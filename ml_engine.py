import os
import pickle
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from historical_db import PREDS_CSV_PATH

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_model.pkl")

FEATURES = ["fair_prob", "ev_pct", "odds", "llm_confidence", "cqs"]

def train_ml_model() -> bool:
    if not os.path.exists(PREDS_CSV_PATH):
        logger.warning("ml_engine: Historical database not found. Cannot train model.")
        print(f"❌ Database historis tidak ditemukan di {PREDS_CSV_PATH}")
        return False
        
    try:
        df = pd.read_csv(PREDS_CSV_PATH)
    except Exception as e:
        logger.error("ml_engine: Failed to read database: %s", e)
        print(f"❌ Gagal membaca database: {e}")
        return False
        
    if df.empty:
        logger.warning("ml_engine: Historical database is empty.")
        print("❌ Database historis kosong.")
        return False
        
    # Filter only WIN and LOSS results
    train_df = df[df["result"].isin(["WIN", "LOSS"])].copy()
    if len(train_df) < 15: # minimum matches to train
        logger.warning("ml_engine: Not enough settled matches to train model (need at least 15, got %d).", len(train_df))
        print(f"❌ Data laga selesai belum cukup untuk latihan model (butuh min 15, ada {len(train_df)}).")
        return False
        
    # Prepare labels: WIN = 1, LOSS = 0
    y = (train_df["result"] == "WIN").astype(int).values
    
    # Prepare features
    X_df = train_df[FEATURES].copy()
    
    # Fill missing values with median
    for col in FEATURES:
        if X_df[col].isnull().any():
            median_val = X_df[col].median()
            if pd.isna(median_val):
                median_val = 0.5 if col in ("fair_prob", "llm_confidence") else 0.0
            X_df[col] = X_df[col].fillna(median_val)
            
    X = X_df.values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Logistic Regression (naturally calibrated probability output)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Save scaler and model
    model_data = {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "intercept": model.intercept_[0],
        "coefficients": model.coef_[0].tolist(),
        "trained_at": datetime.now().isoformat(),
        "data_count": len(train_df)
    }
    
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_data, f)
        logger.info("ml_engine: Model trained and saved successfully to %s", MODEL_PATH)
        print(f"Model ML berhasil dilatih pada {len(train_df)} laga dan disimpan ke {MODEL_PATH}!")
        
        # Display Feature Importance
        print("\n BOBOT KEPUTUSAN FITUR (Feature Importance coefficients):")
        coefs = model.coef_[0]
        importance_rows = []
        for feat, coef in zip(FEATURES, coefs):
            direction = "Meningkatkan Akurasi" if coef > 0 else "Menurunkan Akurasi"
            importance_rows.append([feat, f"{coef:+.3f}", direction])
        from tabulate import tabulate
        print(tabulate(importance_rows, headers=["Nama Fitur", "Bobot (Coefficient)", "Pengaruh terhadap Prediksi Win"], tablefmt="simple"))
        return True
    except Exception as e:
        logger.error("ml_engine: Failed to save model: %s", e)
        print(f"❌ Gagal menyimpan model: {e}")
        return False

def load_ml_model() -> Optional[Dict[str, Any]]:
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("ml_engine: Failed to load model: %s", e)
        return None

def predict_ml_probability(features: Dict[str, float]) -> Optional[Tuple[float, Dict[str, float]]]:
    """Predict calibrated probability of a win and return feature contributions (explainability)."""
    model_data = load_ml_model()
    if not model_data:
        return None
        
    model = model_data["model"]
    scaler = model_data["scaler"]
    features_list = model_data["features"]
    
    # Prepare input vector
    x_input = []
    for feat in features_list:
        val = features.get(feat)
        if val is None or np.isnan(val):
            # Default value falls back to standard defaults
            val = 0.5 if feat in ("fair_prob", "llm_confidence") else 0.0
        x_input.append(val)
        
    x_arr = np.array([x_input])
    
    try:
        # Scale input
        x_scaled = scaler.transform(x_arr)
        
        # Predict probability
        prob = float(model.predict_proba(x_scaled)[0, 1])
        
        # Calculate feature contributions
        # Contribution of scaled_feature_i = scaled_value_i * coefficient_i
        coefs = model.coef_[0]
        scaled_vals = x_scaled[0]
        
        contributions = {}
        total_abs_contrib = sum(abs(v * c) for v, c in zip(scaled_vals, coefs))
        
        for feat, val, coef in zip(features_list, scaled_vals, coefs):
            contrib_val = val * coef
            # Convert contribution to percentage proportion of the prediction adjustment
            # Positive contribution increases final probability, negative decreases it
            contributions[feat] = contrib_val
            
        return prob, contributions
    except Exception as e:
        logger.error("ml_engine: Prediction error: %s", e)
        return None
        
from datetime import datetime
