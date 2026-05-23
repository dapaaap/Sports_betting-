
import os
from dataclasses import dataclass, field
from typing import List, Optional


def _load_api_keys():
    keys = {}
    try:
        import api_keys as _k

        keys["ODDS_PROVIDER"] = getattr(_k, "ODDS_PROVIDER", None)
        keys["THE_ODDS_API_KEY"] = getattr(_k, "THE_ODDS_API_KEY", None)
        keys["ODDS_API_IO_KEY"] = getattr(_k, "ODDS_API_IO_KEY", None)
        keys["GOOGLE_AI_STUDIO_KEY"] = getattr(_k, "GOOGLE_AI_STUDIO_KEY", None)
        keys["GITHUB_TOKEN"] = getattr(_k, "GITHUB_TOKEN", None)
        keys["OLLAMA_API_KEY"] = getattr(_k, "OLLAMA_API_KEY", None)
    except ImportError:
        pass
    return keys


_api_keys = _load_api_keys()


def _get(name: str, default: str = "") -> str:
    from_file = _api_keys.get(name, "")
    if from_file:
        return from_file
    return os.getenv(name, default)


ODDS_PROVIDER = _get("ODDS_PROVIDER", "the_odds_api").lower().replace("-", "_")
THE_ODDS_API_KEY = _get("THE_ODDS_API_KEY", "")
ODDS_API_IO_KEY = _get("ODDS_API_IO_KEY", "")


@dataclass
class GoogleAIConfig:
    api_key: str = field(default_factory=lambda: _get("GOOGLE_AI_STUDIO_KEY", ""))

    model: str = "gemini-2.0-flash-lite"

    enabled: bool = True

    min_ev_for_llm: float = 2.0

    max_concurrent_requests: int = 1

    delay_between_calls: float = 10.0

    timeout_seconds: int = 30

    temperature: float = 0.1

    max_tokens: int = 600

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key not in (
            "",
            "YOUR_GOOGLE_AI_STUDIO_KEY_HERE",
        )


GOOGLE_AI_CONFIG = GoogleAIConfig()


class OllamaAIConfig:
    def __init__(self):
        self.model = "gemma3:4b"
        self.api_key = _get("OLLAMA_API_KEY", "")
        self.enabled = False
        self.min_ev_for_llm = 2.0
        self.max_concurrent_requests = 1
        self.delay_between_calls = 1.0

    @property
    def host(self) -> str:

        return "https://api.ollama.com" if self.api_key else "http://localhost:11434"

    @property
    def is_configured(self) -> bool:
        return bool(self.model)


OLLAMA_AI_CONFIG = OllamaAIConfig()


AI_PROVIDER = "github"
AI_MODEL = "openai/gpt-4o"
AI_BASE_URL = "https://models.github.ai/inference"


@dataclass
class GitHubAIConfig:
    api_key: str = field(default_factory=lambda: _get("GITHUB_TOKEN", ""))
    model: str = AI_MODEL
    base_url: str = AI_BASE_URL
    enabled: bool = True
    min_ev_for_llm: float = 2.0
    max_concurrent_requests: int = 1
    delay_between_calls: float = 4.5
    timeout_seconds: int = 60
    temperature: float = 0.1
    max_tokens: int = 600

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key) and self.api_key not in ("", "YOUR_GITHUB_TOKEN_HERE")


GITHUB_AI_CONFIG = GitHubAIConfig()


ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_API_IO_BASE_URL = "https://api.odds-api.io/v3"
ODDS_API_IO_BOOKMAKERS = os.getenv("ODDS_API_IO_BOOKMAKERS", "Bet365,Unibet")
ODDS_API_IO_MAX_EVENTS = int(os.getenv("ODDS_API_IO_MAX_EVENTS", "50"))
ODDS_API_REGIONS = "eu,uk,us,au"


ODDS_API_MARKETS = "h2h,totals"
ODDS_API_ODDS_FORMAT = "decimal"


@dataclass
class MarketConfig:
    mode: str = "major"


MARKET_CONFIG = MarketConfig()


FOOTBALL_SPORTS_MAJOR = [
    "soccer_epl",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_spain_la_liga",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_netherlands_eredivisie",
]


FOOTBALL_SPORTS_BROAD = FOOTBALL_SPORTS_MAJOR + [
    "soccer_efl_champ",
    "soccer_portugal_primeira_liga",
    "soccer_turkey_super_league",
    "soccer_belgium_first_div",
    "soccer_scotland_premiership",
    "soccer_austria_football_bundesliga",
    "soccer_denmark_superliga",
    "soccer_norway_eliteserien",
    "soccer_sweden_allsvenskan",
    "soccer_swiss_superleague",
    "soccer_greece_super_league",
    "soccer_czech_liga",
    "soccer_russia_premier_league",
    "soccer_ukraine_premier_league",
    "soccer_croatia_football",
    "soccer_romania_liga_1",
    "soccer_serbia_superliga",
    "soccer_poland_ekstraklasa",
    "soccer_conmebol_libertadores",
    "soccer_brazil_campeonato",
    "soccer_usa_mls",
    "soccer_argentina_primera_division",
    "soccer_mexico_ligamx",
    "soccer_chile_primera_division",
    "soccer_colombia_primera_a",
    "soccer_japan_j_league",
    "soccer_south_korea_kleague1",
    "soccer_australia_aleague",
    "soccer_china_superleague",
]


@dataclass
class AnalysisConfig:
    min_edge_pct: float = 3.0

    max_edge_pct: float = 15.0

    min_implied_prob: float = 0.22

    max_implied_prob: float = 0.92

    kelly_fraction: float = 0.5

    max_kelly_pct: float = 3.0

    max_total_stake_per_session: float = 150000.0

    min_bookmakers: int = 4

    lookahead_hours: int = 12

    ai_confidence_threshold: float = 0.62

    min_odds: float = 1.40

    max_odds: float = 4.50

    bankroll: float = 1000000.0

    cqs_min_display: float = 45.0
    cqs_premium_threshold: float = 70.0
    cqs_standard_threshold: float = 55.0

    max_picks_per_session: int = 8

    max_picks_per_league: int = 2


ANALYSIS_CONFIG = AnalysisConfig()


BOOKMAKER_TRUST = {
    "pinnacle": 1.00,
    "betfair_ex_eu": 0.97,
    "matchbook": 0.95,
    "betsson": 0.88,
    "unibet_eu": 0.87,
    "williamhill": 0.85,
    "bet365": 0.85,
    "bwin": 0.82,
    "betway": 0.80,
    "1xbet": 0.78,
    "betonlineag": 0.75,
    "draftkings": 0.80,
    "fanduel": 0.80,
    "mybookieag": 0.70,
    "pinnacle_sports": 1.00,
}


LEAGUE_QUALITY_MULTIPLIER: dict = {
    "soccer_epl": 1.00,
    "soccer_spain_la_liga": 1.00,
    "soccer_germany_bundesliga": 1.00,
    "soccer_uefa_champs_league": 0.95,
    "soccer_italy_serie_a": 0.95,
    "soccer_france_ligue_one": 0.90,
    "soccer_efl_champ": 0.80,
    "soccer_netherlands_eredivisie": 0.85,
    "soccer_portugal_primeira_liga": 0.80,
    "soccer_turkey_super_league": 0.75,
    "soccer_belgium_first_div": 0.80,
    "soccer_uefa_europa_league": 0.80,
    "soccer_sweden_allsvenskan": 0.65,
    "soccer_austria_football_bundesliga": 0.65,
    "soccer_denmark_superliga": 0.65,
    "soccer_norway_eliteserien": 0.65,
    "soccer_swiss_superleague": 0.65,
    "soccer_poland_ekstraklasa": 0.60,
    "soccer_scotland_premiership": 0.65,
    "soccer_brazil_campeonato": 0.55,
    "soccer_usa_mls": 0.55,
    "soccer_mexico_ligamx": 0.55,
    "soccer_conmebol_libertadores": 0.55,
    "soccer_italy_serie_b": 0.50,
    "soccer_greece_super_league": 0.50,
    "soccer_czech_liga": 0.50,
    "soccer_ukraine_premier_league": 0.50,
    "soccer_romania_liga_1": 0.50,
    "soccer_croatia_football": 0.50,
    "soccer_serbia_superliga": 0.50,
    "soccer_argentina_primera_division": 0.40,
    "soccer_saudi_premier_league": 0.40,
    "soccer_russia_premier_league": 0.40,
    "soccer_chile_primera_division": 0.40,
    "soccer_colombia_primera_a": 0.40,
    "soccer_japan_j_league": 0.45,
    "soccer_south_korea_kleague1": 0.45,
    "soccer_australia_aleague": 0.45,
    "soccer_china_superleague": 0.40,
}


LEAGUE_QUALITY_DEFAULT_MULTIPLIER: float = 0.40


LEAGUE_MIN_EDGE: dict = {
    1.00: 3.0,
    0.95: 3.0,
    0.90: 3.0,
    0.85: 3.5,
    0.80: 3.5,
    0.75: 3.5,
    0.65: 4.5,
    0.60: 4.5,
    0.55: 5.5,
    0.50: 5.5,
    0.45: 7.0,
    0.40: 7.0,
}


def get_league_multiplier(sport_key: str) -> float:
    return LEAGUE_QUALITY_MULTIPLIER.get(sport_key, LEAGUE_QUALITY_DEFAULT_MULTIPLIER)


def get_league_min_edge(sport_key: str) -> float:
    mult = get_league_multiplier(sport_key)

    best_edge = ANALYSIS_CONFIG.min_edge_pct
    best_diff = float("inf")
    for tier_mult, min_edge in LEAGUE_MIN_EDGE.items():
        diff = abs(tier_mult - mult)
        if diff < best_diff:
            best_diff = diff
            best_edge = min_edge
    return best_edge


LEAGUE_CONFIDENCE: dict = {
    "english premier league": 1.00,
    "epl": 1.00,
    "spain la liga": 1.00,
    "la liga": 1.00,
    "germany bundesliga": 1.00,
    "bundesliga": 1.00,
    "italy serie a": 0.90,
    "serie a": 0.90,
    "france ligue 1": 0.90,
    "ligue 1": 0.90,
    "uefa champions league": 0.90,
    "champions league": 0.90,
    "uefa europa league": 0.85,
    "netherlands eredivisie": 0.85,
    "eredivisie": 0.85,
    "portugal primeira liga": 0.80,
    "turkey super lig": 0.80,
    "scotland premiership": 0.75,
    "spl": 0.75,
    "efl championship": 0.80,
    "championship": 0.80,
    "italy serie b": 0.70,
    "serie b": 0.70,
    "france ligue 2": 0.70,
    "ligue 2": 0.70,
    "greece super league": 0.65,
    "greek super league": 0.65,
    "sweden superettan": 0.70,
    "sweden allsvenskan": 0.75,
    "norway eliteserien": 0.70,
    "denmark superliga": 0.75,
    "austria bundesliga": 0.70,
    "switzerland super league": 0.70,
    "argentina primera division": 0.60,
    "liga argentina": 0.60,
    "brazil serie a": 0.65,
    "mls": 0.65,
    "saudi arabia pro league": 0.65,
    "saudi pro league": 0.65,
    "j league": 0.65,
    "k league": 0.65,
}


def get_league_confidence(sport_name: str) -> float:
    key = sport_name.lower().strip()

    if key in LEAGUE_CONFIDENCE:
        return LEAGUE_CONFIDENCE[key]

    for league_key, mult in LEAGUE_CONFIDENCE.items():
        if league_key in key or key in league_key:
            return mult
    return 0.70


REFRESH_INTERVAL_SECONDS = 60
MAX_BETS_DISPLAYED = 25
SHOW_ALL_MARKETS = True


COLOR_STRONG_VALUE = "bright_green"
COLOR_GOOD_VALUE = "green"
COLOR_WEAK_VALUE = "yellow"
COLOR_NO_VALUE = "white"
COLOR_HEADER = "bright_cyan"
COLOR_DANGER = "red"
COLOR_INFO = "bright_blue"
