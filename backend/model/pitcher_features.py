"""
Pitcher feature engineering for the v2 game prediction model.

Season stats source : FanGraphs via pybaseball (6 features per SP)
Rolling stats source: MLB Stats API game-by-game logs  (4 features per SP)
Probable starters   : MLB Stats API /schedule?hydrate=probablePitcher

Feature vector per pitcher (10 total):
  [era, whip, k_per_9, bb_per_9, hr_per_9, fip,
   roll_era, roll_whip, roll_k_per_9, roll_avg_ip]

Defaults are MLB league averages when a pitcher is unavailable.
"""

import unicodedata
import numpy as np
import pandas as pd
import requests

MLB_API = "https://statsapi.mlb.com/api/v1"

_SEASON_COLS   = ["ERA", "WHIP", "K/9", "BB/9", "HR/9", "FIP"]
_SEASON_DEFS   = [4.30,   1.28,   8.6,   3.2,    1.15,  4.20]   # 2023-25 MLB avg

_ROLLING_COLS  = ["roll_era", "roll_whip", "roll_k9", "roll_avg_ip"]
_ROLLING_DEFS  = [4.30,        1.28,         8.6,       5.2]

N_SP_FEATS = len(_SEASON_COLS) + len(_ROLLING_COLS)   # 10


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return " ".join(name.lower().split())


# ---------------------------------------------------------------------------
# Season stats (FanGraphs via pybaseball)
# ---------------------------------------------------------------------------

def get_pitcher_season_stats(year: int, through_date: str | None = None) -> pd.DataFrame:
    """
    Return pitcher stats accumulated up to (but not including) through_date.

    through_date: 'YYYY-MM-DD' — stats are fetched for season-start → day before.
                  If None, returns full-season stats (for live serving only — do NOT
                  use None during model training as it leaks future information).

    Returns DataFrame indexed by normalized pitcher name with columns:
      era, whip, k_per_9, bb_per_9, hr_per_9, fip
    """
    try:
        import pybaseball
        pybaseball.cache.enable()
        if through_date is not None:
            # Point-in-time: stats through the day before the game
            season_start = f"{year}-03-01"
            end = (pd.Timestamp(through_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if end < season_start:
                return pd.DataFrame()
            raw = pybaseball.pitching_stats_range(season_start, end)
        else:
            raw = pybaseball.pitching_stats(year, qual=1)
    except Exception:
        return pd.DataFrame()

    needed = ["Name"] + _SEASON_COLS
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        return pd.DataFrame()

    df = raw[needed].copy()
    df.columns = ["name", "era", "whip", "k_per_9", "bb_per_9", "hr_per_9", "fip"]
    df["name"] = df["name"].apply(_norm)
    df = df.drop_duplicates("name").set_index("name")
    return df.apply(pd.to_numeric, errors="coerce")


def pitcher_season_vector(name: str | None, stats: pd.DataFrame) -> np.ndarray:
    """6-element array of season stats. Falls back to league-average defaults."""
    defaults = np.array(_SEASON_DEFS, dtype=float)
    if name is None or stats.empty:
        return defaults

    key = _norm(name)

    # Exact match
    if key in stats.index:
        vals = stats.loc[key].values.astype(float)
        return np.where(np.isnan(vals), defaults, vals)

    # Last-name-only fallback (handles middle initials / accents)
    last = key.split()[-1]
    matches = [i for i in stats.index if i.split()[-1] == last]
    if len(matches) == 1:
        vals = stats.loc[matches[0]].values.astype(float)
        return np.where(np.isnan(vals), defaults, vals)

    return defaults


# ---------------------------------------------------------------------------
# Rolling last-N-start stats (MLB Stats API game logs)
# ---------------------------------------------------------------------------

def _get_pitcher_game_logs(mlbam_id: int, year: int) -> pd.DataFrame:
    """
    Fetch game-by-game pitching log for a player from the MLB Stats API.
    Returns DataFrame with columns: date, ip, era, whip, k, bb
    """
    url = f"{MLB_API}/people/{mlbam_id}/stats"
    params = {
        "stats": "gameLog",
        "group": "pitching",
        "season": year,
        "sportId": 1,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        splits = resp.json()["stats"][0]["splits"]
    except Exception:
        return pd.DataFrame()

    records = []
    for s in splits:
        stat = s.get("stat", {})
        try:
            ip = _ip_str_to_float(stat.get("inningsPitched", "0"))
            records.append({
                "date":  s.get("date", ""),
                "ip":    ip,
                "er":    float(stat.get("earnedRuns", 0)),
                "h":     float(stat.get("hits", 0)),
                "bb":    float(stat.get("baseOnBalls", 0)),
                "k":     float(stat.get("strikeOuts", 0)),
                "hr":    float(stat.get("homeRuns", 0)),
            })
        except (ValueError, TypeError):
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _ip_str_to_float(ip_str) -> float:
    """'6.2' in baseball notation = 6⅔ innings = 6.667 actual innings."""
    try:
        parts = str(ip_str).split(".")
        full = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return full + thirds / 3
    except (ValueError, IndexError):
        return 0.0


def _rolling_stats_from_logs(logs: pd.DataFrame, n: int = 5) -> np.ndarray:
    """Compute rolling-last-N-start stats from a game log DataFrame."""
    defaults = np.array(_ROLLING_DEFS, dtype=float)
    if logs.empty:
        return defaults

    recent = logs.tail(n)
    total_ip = recent["ip"].sum()
    if total_ip < 0.1:
        return defaults

    era  = recent["er"].sum() / total_ip * 9
    ha   = recent["h"].sum() + recent["bb"].sum()
    whip = ha / total_ip
    k9   = recent["k"].sum() / total_ip * 9
    avg_ip = total_ip / len(recent)

    result = np.array([era, whip, k9, avg_ip], dtype=float)
    return np.where(np.isnan(result) | np.isinf(result), defaults, result)


# Cache to avoid repeated API calls during training
_rolling_cache: dict[tuple, np.ndarray] = {}


def pitcher_rolling_vector(name: str | None, year: int, n_starts: int = 5) -> np.ndarray:
    """4-element rolling-last-N-start stat vector."""
    defaults = np.array(_ROLLING_DEFS, dtype=float)
    if name is None:
        return defaults

    cache_key = (_norm(name), year, n_starts)
    if cache_key in _rolling_cache:
        return _rolling_cache[cache_key]

    try:
        import pybaseball
        parts = _norm(name).split()
        if len(parts) < 2:
            return defaults
        ids = pybaseball.playerid_lookup(parts[-1].title(), parts[0].title())
        if ids.empty:
            _rolling_cache[cache_key] = defaults
            return defaults
        mlbam_id = int(ids.iloc[0]["key_mlbam"])
        logs = _get_pitcher_game_logs(mlbam_id, year)
        # Filter to starts only (IP >= 1 is a rough proxy)
        starts = logs[logs["ip"] >= 1.0]
        result = _rolling_stats_from_logs(starts, n=n_starts)
    except Exception:
        result = defaults

    _rolling_cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Combined pitcher feature vector
# ---------------------------------------------------------------------------

def pitcher_feature_vector(
    name: str | None,
    season_stats: pd.DataFrame,
    year: int,
    include_rolling: bool = True,
) -> np.ndarray:
    """Full 10-element pitcher feature vector."""
    season = pitcher_season_vector(name, season_stats)
    if include_rolling:
        rolling = pitcher_rolling_vector(name, year)
    else:
        rolling = np.array(_ROLLING_DEFS, dtype=float)
    return np.concatenate([season, rolling])


def pitcher_feature_columns() -> list[str]:
    season = ["era", "whip", "k_per_9", "bb_per_9", "hr_per_9", "fip"]
    rolling = _ROLLING_COLS
    return (
        [f"home_sp_{c}" for c in season + rolling] +
        [f"away_sp_{c}" for c in season + rolling]
    )


# ---------------------------------------------------------------------------
# Probable / actual starters from MLB Stats API
# ---------------------------------------------------------------------------

# MLB Stats API team name → Odds API team name (reuse mapping from results.py)
_MLB_TO_ODDS = {
    "Arizona Diamondbacks": "Arizona Diamondbacks",
    "Atlanta Braves": "Atlanta Braves",
    "Baltimore Orioles": "Baltimore Orioles",
    "Boston Red Sox": "Boston Red Sox",
    "Chicago Cubs": "Chicago Cubs",
    "Chicago White Sox": "Chicago White Sox",
    "Cincinnati Reds": "Cincinnati Reds",
    "Cleveland Guardians": "Cleveland Guardians",
    "Colorado Rockies": "Colorado Rockies",
    "Detroit Tigers": "Detroit Tigers",
    "Houston Astros": "Houston Astros",
    "Kansas City Royals": "Kansas City Royals",
    "Los Angeles Angels": "Los Angeles Angels",
    "Los Angeles Dodgers": "Los Angeles Dodgers",
    "Miami Marlins": "Miami Marlins",
    "Milwaukee Brewers": "Milwaukee Brewers",
    "Minnesota Twins": "Minnesota Twins",
    "New York Mets": "New York Mets",
    "New York Yankees": "New York Yankees",
    "Oakland Athletics": "Athletics",
    "Philadelphia Phillies": "Philadelphia Phillies",
    "Pittsburgh Pirates": "Pittsburgh Pirates",
    "San Diego Padres": "San Diego Padres",
    "San Francisco Giants": "San Francisco Giants",
    "Seattle Mariners": "Seattle Mariners",
    "St. Louis Cardinals": "St. Louis Cardinals",
    "Tampa Bay Rays": "Tampa Bay Rays",
    "Texas Rangers": "Texas Rangers",
    "Toronto Blue Jays": "Toronto Blue Jays",
    "Washington Nationals": "Washington Nationals",
    "Athletics": "Athletics",
}


def get_starters_for_date(date_str: str) -> dict[str, dict]:
    """
    Fetch probable/actual starting pitchers for all games on a date.

    Returns:
      {
        "New York Yankees vs Boston Red Sox": {
          "home_team": "New York Yankees",
          "away_team": "Boston Red Sox",
          "home_pitcher": "Gerrit Cole",
          "away_pitcher": "Chris Sale",
          "home_pitcher_id": 543037,
          "away_pitcher_id": 519242,
        },
        ...
      }
    Keyed by "home_team vs away_team" for easy lookup.
    """
    url = f"{MLB_API}/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception:
        return {}

    result = {}
    for date_entry in resp.json().get("dates", []):
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            home_mlb = teams.get("home", {}).get("team", {}).get("name", "")
            away_mlb = teams.get("away", {}).get("team", {}).get("name", "")
            home_odds = _MLB_TO_ODDS.get(home_mlb, home_mlb)
            away_odds = _MLB_TO_ODDS.get(away_mlb, away_mlb)
            home_p = teams.get("home", {}).get("probablePitcher") or {}
            away_p = teams.get("away", {}).get("probablePitcher") or {}
            key = f"{home_odds} vs {away_odds}"
            result[key] = {
                "home_team":       home_odds,
                "away_team":       away_odds,
                "home_pitcher":    home_p.get("fullName"),
                "away_pitcher":    away_p.get("fullName"),
                "home_pitcher_id": home_p.get("id"),
                "away_pitcher_id": away_p.get("id"),
            }
    return result
