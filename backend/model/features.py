"""
Feature engineering for game-level MLB win prediction.

Uses pybaseball to pull team batting and pitching stats for the current season.
Features per game:
  - Home team: batting avg, OBP, SLG, ERA, WHIP, win%
  - Away team: same
  - Home field advantage indicator (always 1, kept for model consistency)

Team name normalization maps The Odds API names -> pybaseball FanGraphs names.
"""

import pandas as pd
import numpy as np
from datetime import datetime

try:
    import pybaseball
    pybaseball.cache.enable()
    PYBASEBALL_AVAILABLE = True
except ImportError:
    PYBASEBALL_AVAILABLE = False

# Odds API team name -> FanGraphs team abbreviation
TEAM_NAME_TO_ABB = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
    # common alternate spellings
    "Athletics": "OAK",
    "Guardians": "CLE",
}

_BATTING_COLS = ["Team", "AVG", "OBP", "SLG", "BB%", "K%"]
_PITCHING_COLS = ["Team", "ERA", "WHIP", "K/9", "BB/9", "HR/9", "FIP"]


def _get_season_batting(year: int) -> pd.DataFrame:
    df = pybaseball.team_batting(year)
    df = df[_BATTING_COLS].copy()
    df.columns = ["team_abb", "bat_avg", "bat_obp", "bat_slg", "bat_bb_pct", "bat_k_pct"]
    return df.set_index("team_abb")


def _get_season_pitching(year: int) -> pd.DataFrame:
    df = pybaseball.team_pitching(year)
    df = df[_PITCHING_COLS].copy()
    df.columns = ["team_abb", "pit_era", "pit_whip", "pit_k9", "pit_bb9", "pit_hr9", "pit_fip"]
    return df.set_index("team_abb")


def build_team_stats(year: int | None = None) -> pd.DataFrame:
    """Return a merged batting + pitching DataFrame indexed by team abbreviation."""
    if not PYBASEBALL_AVAILABLE:
        return pd.DataFrame()
    year = year or datetime.now().year
    batting = _get_season_batting(year)
    pitching = _get_season_pitching(year)
    return batting.join(pitching, how="inner")


def game_features(home_team: str, away_team: str, stats: pd.DataFrame) -> np.ndarray | None:
    """
    Build a feature vector for a single matchup.
    Returns None if team data isn't available.
    """
    home_abb = TEAM_NAME_TO_ABB.get(home_team)
    away_abb = TEAM_NAME_TO_ABB.get(away_team)

    if home_abb is None or away_abb is None:
        return None
    if stats.empty or home_abb not in stats.index or away_abb not in stats.index:
        return None

    home = stats.loc[home_abb].values.astype(float)
    away = stats.loc[away_abb].values.astype(float)
    # [home_features..., away_features..., home_field_advantage]
    return np.concatenate([home, away, [1.0]])


def feature_columns() -> list[str]:
    batting_feats = ["bat_avg", "bat_obp", "bat_slg", "bat_bb_pct", "bat_k_pct"]
    pitching_feats = ["pit_era", "pit_whip", "pit_k9", "pit_bb9", "pit_hr9", "pit_fip"]
    cols = (
        [f"home_{c}" for c in batting_feats + pitching_feats]
        + [f"away_{c}" for c in batting_feats + pitching_feats]
        + ["home_field"]
    )
    return cols
