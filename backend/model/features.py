"""
Feature engineering for game-level MLB win prediction.

Per-game batting logs are read from the parquet files in
artifacts/game_log_data/ (populated by update_game_logs.py).
Rolling pre-game stats (last ROLLING_WINDOW games) are computed so that
no future data leaks into any game's feature vector.

Features (15 total):
  - Home team: rpg, rapg, obp, slg, bb_pct, k_pct, wpct  (7)
  - Away team: same                                         (7)
  - home_field = 1                                         (1)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Odds API team name -> BRef abbreviation
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
    "Athletics": "OAK",
    "Guardians": "CLE",
}

ROLLING_WINDOW = 15   # games used for rolling pre-game stats
MIN_GAMES = 10        # minimum prior games required for a training row

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "game_log_data")
_ATH_YEAR = 2025      # BRef switched OAK → ATH from this season on


def _parquet_path(team_abb: str, year: int) -> str:
    file_abb = "ATH" if team_abb == "OAK" and year >= _ATH_YEAR else team_abb
    return os.path.join(_ARTIFACTS_DIR, f"batting_{file_abb}_{year}_logs.parquet")


def fetch_batting_game_log(year: int, team_abb: str) -> pd.DataFrame:
    """
    Load the per-game batting log for one team-season from the local parquet.
    Returns a DataFrame sorted by date with columns:
      Date, Home (bool), Opp, Rslt, win, RS, RA, PA, AB, H, 2B, 3B, HR,
      BB, SO, HBP, SF, TB, OBP, SLG
    Returns empty DataFrame if the parquet does not exist.
    """
    path = _parquet_path(team_abb, year)
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "RS", "RA"])

        if "Rslt" in df.columns:
            df["win"] = df["Rslt"].astype(str).str.startswith("W").astype(int)

        return df.sort_values("Date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _rolling_team_stats(window_df: pd.DataFrame) -> dict:
    """Compute aggregate stats from a window of completed games."""
    n = len(window_df)
    if n == 0:
        return {k: np.nan for k in ["rpg", "rapg", "obp", "slg", "bb_pct", "k_pct"]}

    rpg  = window_df["RS"].sum() / n
    rapg = window_df["RA"].sum() / n

    ab  = window_df["AB"].sum()
    pa  = window_df["PA"].sum()
    h   = window_df["H"].sum()
    bb  = window_df["BB"].sum()
    so  = window_df["SO"].sum()
    hbp = window_df["HBP"].fillna(0).sum() if "HBP" in window_df else 0
    sf  = window_df["SF"].fillna(0).sum()  if "SF"  in window_df else 0
    tb  = window_df["TB"].sum() if "TB" in window_df else np.nan

    denom_obp = ab + bb + hbp + sf
    obp    = (h + bb + hbp) / denom_obp if denom_obp > 0 else np.nan
    slg    = tb / ab if (not np.isnan(tb) and ab > 0) else np.nan
    bb_pct = bb / pa if pa > 0 else np.nan
    k_pct  = so / pa if pa > 0 else np.nan

    return {"rpg": rpg, "rapg": rapg, "obp": obp,
            "slg": slg, "bb_pct": bb_pct, "k_pct": k_pct}


def build_team_game_lookup(year: int) -> dict:
    """
    For every team load their batting game log and compute rolling
    pre-game stats for each game in the season.

    Returns:
        {team_abb: DataFrame with columns
            [Date, Home, Opp, win, rpg, rapg, obp, slg, bb_pct, k_pct, wpct, n_prior]}
        where every stat column reflects the team's state *before* that game.
    """
    lookup = {}
    for abb in sorted(set(TEAM_NAME_TO_ABB.values())):
        log = fetch_batting_game_log(year, abb)
        if log.empty or "win" not in log.columns:
            continue

        pre_stats = []
        cumulative_wins = 0
        cumulative_games = 0

        for i, row in log.iterrows():
            window = log.iloc[max(0, i - ROLLING_WINDOW):i]
            stats = _rolling_team_stats(window)
            stats["wpct"] = (cumulative_wins / cumulative_games
                             if cumulative_games > 0 else np.nan)
            stats["n_prior"] = i
            pre_stats.append(stats)

            cumulative_wins  += int(row["win"])
            cumulative_games += 1

        stats_df = pd.DataFrame(pre_stats)
        result = pd.concat([
            log[["Date", "Home", "Opp", "win"]].reset_index(drop=True),
            stats_df.reset_index(drop=True),
        ], axis=1)
        lookup[abb] = result

    return lookup


def build_team_stats(year=None) -> pd.DataFrame:
    """
    Return current-season rolling stats for all teams (last ROLLING_WINDOW games),
    indexed by team abbreviation. Used by GamePredictor for live predictions.
    """
    year = year or datetime.now().year
    rows = {}
    for abb in sorted(set(TEAM_NAME_TO_ABB.values())):
        log = fetch_batting_game_log(year, abb)
        if log.empty or len(log) == 0:
            continue
        stats = _rolling_team_stats(log.tail(ROLLING_WINDOW))
        stats["wpct"] = float(log["win"].mean()) if "win" in log.columns else np.nan
        rows[abb] = stats
    return pd.DataFrame(rows).T


def game_features(home_team: str, away_team: str, stats: pd.DataFrame):
    """
    Build a feature vector for a single matchup from a stats DataFrame
    (as returned by build_team_stats). Returns None if data is missing.
    """
    home_abb = TEAM_NAME_TO_ABB.get(home_team)
    away_abb = TEAM_NAME_TO_ABB.get(away_team)

    if home_abb is None or away_abb is None:
        return None
    if stats.empty or home_abb not in stats.index or away_abb not in stats.index:
        return None

    _team_feats = ["rpg", "rapg", "obp", "slg", "bb_pct", "k_pct", "wpct"]
    home = stats.loc[home_abb, _team_feats].values.astype(float)
    away = stats.loc[away_abb, _team_feats].values.astype(float)
    return np.concatenate([home, away, [1.0]])


def feature_columns() -> list:
    team_feats = ["rpg", "rapg", "obp", "slg", "bb_pct", "k_pct", "wpct"]
    return (
        [f"home_{c}" for c in team_feats]
        + [f"away_{c}" for c in team_feats]
        + ["home_field"]
    )
