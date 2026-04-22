"""
Feature engineering for game-level MLB win prediction.

Per-game batting logs from Baseball Reference are fetched for each team.
Rolling pre-game stats (last ROLLING_WINDOW games) are computed so that
no future data leaks into any game's feature vector.

Features (15 total):
  - Home team: rpg, rapg, obp, slg, bb_pct, k_pct, wpct  (7)
  - Away team: same                                         (7)
  - home_field = 1                                         (1)
"""

import io
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False

try:
    import pybaseball
    pybaseball.cache.enable()
    from pybaseball.datasources.bref import BRefSession
    _session = BRefSession()
    PYBASEBALL_AVAILABLE = True
except Exception:
    PYBASEBALL_AVAILABLE = False

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
MIN_GAMES = 10        # minimum prior games required to include a row in training

_BREF_LOG_URL = (
    "https://www.baseball-reference.com/teams/tgl.cgi"
    "?team={team}&t=b&year={year}"
)

_NUMERIC_COLS = ["RS", "RA", "Inn", "PA", "AB", "R", "H",
                 "2B", "3B", "HR", "RBI", "BB", "SO",
                 "HBP", "SH", "SF", "TB", "OBP", "SLG"]


def fetch_batting_game_log(year: int, team_abb: str) -> pd.DataFrame:
    """
    Pull the per-game batting log for one team-season from Baseball Reference.
    Returns a DataFrame sorted by date with columns:
      Date, Home (bool), Opp, Rslt, win, RS, RA, PA, AB, H, 2B, 3B, HR,
      BB, SO, HBP, SF, TB, OBP, SLG
    Returns empty DataFrame on any failure.
    """
    if not (PYBASEBALL_AVAILABLE and _BS4):
        return pd.DataFrame()
    try:
        url = _BREF_LOG_URL.format(team=team_abb, year=year)
        content = _session.get(url).content
        soup = BeautifulSoup(content, "html.parser")
        table = soup.find("table", {"id": "players_standard_batting"})
        if table is None:
            return pd.DataFrame()

        df = pd.read_html(io.StringIO(str(table)))[0]

        # Flatten multi-level column headers produced by BRef tables
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        # Drop rank column
        df = df.drop(columns=[c for c in df.columns if str(c) == "Rk"], errors="ignore")

        # Rename game-number column
        df = df.rename(columns={"Gtm": "Game"})

        # The home/away indicator column is unnamed; identify it by position
        # It contains '@' for away games and is blank for home games
        home_col = next(
            (c for c in df.columns if "Unnamed" in str(c) and "3_level" in str(c)), None
        )
        if home_col:
            df["Home"] = df[home_col].isna() | (df[home_col].astype(str).str.strip() == "")
            df = df.drop(columns=[home_col])
        elif "Home" not in df.columns:
            df["Home"] = True

        # Keep only completed game rows (Game column is numeric)
        game_col = df.get("Game", pd.Series(dtype=str))
        df = df[pd.to_numeric(game_col, errors="coerce").notna()].copy()

        # Parse date and numeric columns
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Derive win flag from result column
        rslt_col = "Rslt" if "Rslt" in df.columns else None
        if rslt_col:
            df["win"] = df[rslt_col].astype(str).str.startswith("W").astype(int)

        df = df.dropna(subset=["Date", "RS", "RA"])
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
    For every team fetch their batting game log and compute rolling
    pre-game stats for each game in the season.

    Returns:
        {team_abb: DataFrame with columns
            [Date, Home, Opp, win, rpg, rapg, obp, slg, bb_pct, k_pct, wpct]}
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
            # Stats computed from games played BEFORE this one
            window = log.iloc[max(0, i - ROLLING_WINDOW):i]
            stats = _rolling_team_stats(window)
            stats["wpct"] = (cumulative_wins / cumulative_games
                             if cumulative_games > 0 else np.nan)
            stats["n_prior"] = i  # number of prior games this season
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
    Return current-season rolling stats for all teams (season-to-date),
    indexed by team abbreviation.  Used by GamePredictor for live predictions.
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
    (as returned by build_team_stats).  Returns None if data is missing.
    """
    home_abb = TEAM_NAME_TO_ABB.get(home_team)
    away_abb = TEAM_NAME_TO_ABB.get(away_team)

    if home_abb is None or away_abb is None:
        return None
    if stats.empty or home_abb not in stats.index or away_abb not in stats.index:
        return None

    home = stats.loc[home_abb, feature_columns()[:7]].values.astype(float)
    away = stats.loc[away_abb, feature_columns()[:7]].values.astype(float)
    return np.concatenate([home, away, [1.0]])


def feature_columns() -> list:
    team_feats = ["rpg", "rapg", "obp", "slg", "bb_pct", "k_pct", "wpct"]
    return (
        [f"home_{c}" for c in team_feats]
        + [f"away_{c}" for c in team_feats]
        + ["home_field"]
    )
