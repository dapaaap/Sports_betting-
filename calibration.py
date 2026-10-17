import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

CALIBRATION_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "calibration_model.pkl"
)

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) calculation."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_prob)
    if n_samples == 0:
        return 0.0
        
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Include upper boundary in the last bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += (bin_size / n_samples) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)

def calculate_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Clips probabilities to avoid numerical instability."""
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    loss = - (y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(np.mean(loss))

def calculate_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score = Mean Squared Error of probabilities."""
    return float(np.mean((y_true - y_prob) ** 2))


class CalibrationLayer:
    def __init__(self):
        self.platt_model: Optional[LogisticRegression] = None
        self.isotonic_model: Optional[IsotonicRegression] = None
        self.is_fitted = False
        self.load_model()

    def train(self, raw_probs: List[float], outcomes: List[int]) -> bool:
        """
        Fits Platt Scaling and Isotonic Regression calibration models.
        outcomes list must contain 1 for WIN / Success, 0 for LOSS / Failure.
        """
        X = np.array(raw_probs).reshape(-1, 1)
        y = np.array(outcomes)

        if len(y) < 15:
            logger.warning("CalibrationLayer: Insufficient data to train calibration (%d matches, need at least 15)", len(y))
            return False

        try:
            # Platt Scaling: Logistic Regression on raw probabilities
            platt = LogisticRegression(C=1.0, max_iter=1000)
            platt.fit(X, y)
            self.platt_model = platt

            # Isotonic Regression: non-parametric, monotonic mapping
            isotonic = IsotonicRegression(out_of_bounds="clip")
            isotonic.fit(raw_probs, y)
            self.isotonic_model = isotonic

            self.is_fitted = True
            self.save_model()
            logger.info("CalibrationLayer: Calibration models successfully trained and saved.")
            return True
        except Exception as e:
            logger.error("CalibrationLayer: Failed to train models: %s", e)
            return False

    def calibrate(self, raw_prob: float, method: str = "platt") -> float:
        """Calibrate raw probability."""
        if not self.is_fitted:
            return raw_prob

        # Bounding raw probability to avoid extreme inputs
        raw_prob_val = max(0.001, min(0.999, raw_prob))

        try:
            if method == "platt" and self.platt_model:
                X_input = np.array([[raw_prob_val]])
                # Predict probability of class 1 (win)
                calibrated = float(self.platt_model.predict_proba(X_input)[0, 1])
                return max(0.001, min(0.999, calibrated))
            elif method == "isotonic" and self.isotonic_model:
                calibrated = float(self.isotonic_model.predict([raw_prob_val])[0])
                return max(0.001, min(0.999, calibrated))
        except Exception as e:
            logger.warning("CalibrationLayer: Calibration failed (method=%s), returning raw probability. Error: %s", method, e)

        return raw_prob

    def save_model(self):
        try:
            model_data = {
                "platt": self.platt_model,
                "isotonic": self.isotonic_model,
                "is_fitted": self.is_fitted
            }
            with open(CALIBRATION_MODEL_PATH, "wb") as f:
                pickle.dump(model_data, f)
        except Exception as e:
            logger.error("CalibrationLayer: Failed to save model file: %s", e)

    def load_model(self):
        if not os.path.exists(CALIBRATION_MODEL_PATH):
            return
        try:
            # Security check: verify path is in the same directory as this module
            proj_dir = os.path.dirname(os.path.abspath(__file__))
            abs_model_path = os.path.abspath(CALIBRATION_MODEL_PATH)
            if not abs_model_path.startswith(proj_dir):
                logger.error("CalibrationLayer: Rejected loading pickle from unsafe location: %s", abs_model_path)
                return
                
            # Limit file size to 10MB to avoid large payloads
            if os.path.getsize(CALIBRATION_MODEL_PATH) > 10 * 1024 * 1024:
                logger.error("CalibrationLayer: Model file too large, skipping load.")
                return
                
            with open(CALIBRATION_MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                if not isinstance(data, dict) or "platt" not in data or "isotonic" not in data:
                    logger.error("CalibrationLayer: Loaded pickle lacks required keys, skipping.")
                    return
                self.platt_model = data.get("platt")
                self.isotonic_model = data.get("isotonic")
                self.is_fitted = data.get("is_fitted", False)
        except Exception as e:
            logger.error("CalibrationLayer: Failed to load model file: %s", e)

    def evaluate_calibration(self, raw_probs: List[float], outcomes: List[int]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate and compare Uncalibrated, Platt, and Isotonic Calibration metrics:
        Brier Score, Log Loss, and ECE.
        """
        y_true = np.array(outcomes)
        X_raw = np.array(raw_probs)
        
        # Calculate uncalibrated metrics
        metrics = {
            "uncalibrated": {
                "brier": calculate_brier_score(y_true, X_raw),
                "log_loss": calculate_log_loss(y_true, X_raw),
                "ece": calculate_ece(y_true, X_raw)
            }
        }

        # Calculate Platt calibrated metrics
        if self.is_fitted and self.platt_model:
            platt_probs = np.array([self.calibrate(p, method="platt") for p in raw_probs])
            metrics["platt"] = {
                "brier": calculate_brier_score(y_true, platt_probs),
                "log_loss": calculate_log_loss(y_true, platt_probs),
                "ece": calculate_ece(y_true, platt_probs)
            }

        # Calculate Isotonic calibrated metrics
        if self.is_fitted and self.isotonic_model:
            isotonic_probs = np.array([self.calibrate(p, method="isotonic") for p in raw_probs])
            metrics["isotonic"] = {
                "brier": calculate_brier_score(y_true, isotonic_probs),
                "log_loss": calculate_log_loss(y_true, isotonic_probs),
                "ece": calculate_ece(y_true, isotonic_probs)
            }

        return metrics
