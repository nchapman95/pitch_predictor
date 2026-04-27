"""
Feature engineering for model serving (v1 pipeline).

Replicates the Bayesian / rolling / season-to-date features from
data_pipeline.ipynb using the parquet game logs stored under
artifacts/game_log_data/.

Key public API:
    get_matchup_features(home_team, away_team, final_features, year=None)
        → numpy array matching the ordered list of column names in final_features,
          or None if data is unavailable for either team.
"""

import glob
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from model.features import TEAM_NAME_TO_ABB

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
GAME_LOG_DIR = os.path.join(ARTIFACTS_DIR, "game_log_data")

WINDOW = 10

batting_stat_cols = [
    "PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS",
    "BB", "SO", "BA", "OBP", "SLG", "OPS", "TB", "GIDP", "HBP",
    "SH", "SF", "ROE", "IBB", "BAbip", "LOB",
]
pitching_stat_cols = [
    "IP", "ER", "HR", "BB", "IBB", "SO", "HBP", "BK", "WP", "BF",
    "ERA", "FIP", "Pit", "Str", "StL", "StS", "GB", "FB", "LD",
    "PU", "Unk", "IR", "IS", "PO",
]

# Stats that appear in BOTH batting and pitching — get _home_pit suffix in training
_SHARED = frozenset(batting_stat_cols) & frozenset(pitching_stat_cols)
_PIT_ONLY = frozenset(pitching_stat_cols) - frozenset(batting_stat_cols)


# ---------------------------------------------------------------------------
# Module-level feature cache (refreshed when year changes)
# ---------------------------------------------------------------------------
_cache: dict = {"year": None, "batting": None, "pitching": None}


def _read_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    parts = os.path.basename(path).split("_")
    df["season"] = int(parts[2])
    df["log_type"] = parts[0]
    df["team"] = parts[1].replace("ATH", "OAK")
    return df


def _load_game_logs(years: list) -> pd.DataFrame:
    files = glob.glob(os.path.join(GAME_LOG_DIR, "*.parquet"))
    year_set = set(years)
    dfs = [
        _read_parquet(f)
        for f in files
        if int(os.path.basename(f).split("_")[2]) in year_set
    ]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _compute_std_and_roll(df: pd.DataFrame, stat_cols: list) -> pd.DataFrame:
    available = [c for c in stat_cols if c in df.columns]
    df = df.sort_values(["team", "season", "Date"]).copy()
    grp = df.groupby(["team", "season"])[available]
    std = grp.transform(lambda x: x.shift(1).expanding().mean()).add_suffix("_std")
    roll = grp.transform(
        lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean()
    ).add_suffix(f"_roll{WINDOW}")
    return pd.concat([df, std, roll], axis=1)


def _compute_team_priors(df: pd.DataFrame, stat_cols: list) -> pd.DataFrame:
    available = [c for c in stat_cols if c in df.columns]
    agg = df.groupby(["team", "season"])[available].agg(["mean", "var"]).reset_index()
    agg.columns = ["team", "season"] + [
        f"{col}_{fn}" for col, fn in agg.columns[2:]
    ]
    agg["season"] = agg["season"].astype(int) + 1
    return agg


def _compute_league_priors(df: pd.DataFrame, stat_cols: list) -> pd.DataFrame:
    available = [c for c in stat_cols if c in df.columns]
    agg = df.groupby("season")[available].agg(["mean", "var"]).reset_index()
    agg.columns = ["season"] + [f"{col}_{fn}" for col, fn in agg.columns[1:]]
    return agg


def _bayes_update_series(observations, prior_mean, prior_var):
    """Normal-normal conjugate update; returns (mus, vars_) one step ahead."""
    mus, vars_ = [], []
    mu, var = float(prior_mean), float(prior_var)
    for x in observations:
        mus.append(mu)
        vars_.append(var)
        var_new = 1.0 / (1.0 / var + 1.0 / prior_var)
        mu_new = var_new * (mu / var + float(x) / prior_var)
        mu, var = mu_new, var_new
    return mus, vars_, mu, var  # last mu/var = post-season estimate


def _apply_bayesian(
    df: pd.DataFrame,
    stat_cols: list,
    team_priors: pd.DataFrame,
    league_priors: pd.DataFrame,
) -> pd.DataFrame:
    available = [c for c in stat_cols if c in df.columns]
    results = []

    for (team, season), group in df.sort_values(
        ["team", "season", "Date"]
    ).groupby(["team", "season"]):
        group = group.copy()
        tp = team_priors[
            (team_priors["team"] == team) & (team_priors["season"] == season)
        ]
        lp = league_priors[league_priors["season"] == season]

        new_cols: dict = {}
        for stat in available:
            mean_col, var_col = f"{stat}_mean", f"{stat}_var"

            if not tp.empty and mean_col in tp.columns:
                p_mean = float(tp[mean_col].values[0])
                p_var = float(tp[var_col].values[0])
            elif not lp.empty and mean_col in lp.columns:
                p_mean = float(lp[mean_col].values[0])
                p_var = float(lp[var_col].values[0])
            else:
                p_mean, p_var = 0.0, 1e-6

            if pd.isna(p_var) or p_var <= 0:
                if not lp.empty and var_col in lp.columns:
                    p_var = float(lp[var_col].values[0])
                if pd.isna(p_var) or p_var <= 0:
                    p_var = 1e-6

            obs = pd.to_numeric(group[stat], errors="coerce").fillna(p_mean).values
            mus, vars_, last_mu, last_var = _bayes_update_series(obs, p_mean, p_var)

            new_cols[f"{stat}_bayes_mean"] = mus
            new_cols[f"{stat}_bayes_var"] = vars_
            new_cols[f"{stat}_bayes_mean_current"] = last_mu
            new_cols[f"{stat}_bayes_var_current"] = last_var

        group = pd.concat([group, pd.DataFrame(new_cols, index=group.index)], axis=1)
        results.append(group)

    return pd.concat(results).reset_index(drop=True)


def build_current_features(year=None) -> tuple:
    """
    Load game logs and compute all features for *year*.
    Previous year is used for Bayesian priors.

    Returns: (batting_df, pitching_df) — each indexed by team with feature cols.
    """
    year = int(year or datetime.now().year)
    game_logs = _load_game_logs([year - 1, year])
    if game_logs.empty:
        return pd.DataFrame(), pd.DataFrame()

    batting = game_logs[game_logs["log_type"] == "batting"].copy()
    pitching = game_logs[game_logs["log_type"] == "pitching"].copy()

    # Season-to-date and rolling
    batting = _compute_std_and_roll(batting, batting_stat_cols)
    pitching = _compute_std_and_roll(pitching, pitching_stat_cols)

    # Bayesian priors from previous year
    bat_priors = _compute_team_priors(
        batting[batting["season"] == year - 1], batting_stat_cols
    )
    pit_priors = _compute_team_priors(
        pitching[pitching["season"] == year - 1], pitching_stat_cols
    )
    bat_league = _compute_league_priors(batting, batting_stat_cols)
    pit_league = _compute_league_priors(pitching, pitching_stat_cols)

    batting = _apply_bayesian(batting, batting_stat_cols, bat_priors, bat_league)
    pitching = _apply_bayesian(pitching, pitching_stat_cols, pit_priors, pit_league)

    # Keep only current year
    batting = batting[batting["season"] == year].copy()
    pitching = pitching[pitching["season"] == year].copy()

    return batting, pitching


def _get_cached(year) -> tuple:
    year = int(year)
    if _cache["year"] != year or _cache["batting"] is None:
        _cache["batting"], _cache["pitching"] = build_current_features(year)
        _cache["year"] = year
    return _cache["batting"], _cache["pitching"]


def _latest(df: pd.DataFrame, team: str) -> Optional[pd.Series]:
    rows = df[df["team"] == team].sort_values("Date")
    return rows.iloc[-1] if len(rows) > 0 else None


def _build_feat_dict(
    home_bat: pd.Series,
    home_pit: pd.Series,
    away_bat: pd.Series,
    away_pit: pd.Series,
    year: int,
) -> dict:
    """
    Constructs the feature name → value mapping mirroring data_pipeline.ipynb merges.

    Merge order and suffix rules (from the training notebook):
      1. Home batting  → no suffix   (first merge, no collisions)
      2. Home pitching → no suffix for pitching-only stats;
                         _home_pit suffix for shared stats (collision with batting)
      3. Away batting  → _away_bat suffix for all (collision with home batting)
      4. Away pitching → _away_pit suffix for all (collision with previous merges)
    """
    feat = {}
    suffixes = ["_bayes_mean", "_bayes_var", "_std", f"_roll{WINDOW}"]

    # Use _current variants for Bayesian (post-last-game estimate) so predictions
    # reflect all completed games, not just pre-last-game.
    def _val(row: pd.Series, stat: str, suf: str) -> float:
        if suf == "_bayes_mean" and f"{stat}_bayes_mean_current" in row.index:
            return float(row[f"{stat}_bayes_mean_current"])
        if suf == "_bayes_var" and f"{stat}_bayes_var_current" in row.index:
            return float(row[f"{stat}_bayes_var_current"])
        col = f"{stat}{suf}"
        v = row.get(col, np.nan)
        return float(v) if v is not None else np.nan

    # 1. Home batting
    for stat in batting_stat_cols:
        for suf in suffixes:
            feat[f"{stat}{suf}"] = _val(home_bat, stat, suf)

    # 2. Home pitching
    for stat in pitching_stat_cols:
        for suf in suffixes:
            base = f"{stat}{suf}"
            key = f"{base}_home_pit" if stat in _SHARED else base
            feat[key] = _val(home_pit, stat, suf)

    # 3. Away batting
    for stat in batting_stat_cols:
        for suf in suffixes:
            feat[f"{stat}{suf}_away_bat"] = _val(away_bat, stat, suf)

    # 4. Away pitching
    for stat in pitching_stat_cols:
        for suf in suffixes:
            feat[f"{stat}{suf}_away_pit"] = _val(away_pit, stat, suf)

    # Join-key residuals that land in final_features (numeric, non-bat/pit named)
    feat["season_home_pit"] = float(year)
    feat["season_away_bat"] = float(year)
    feat["season_away_pit"] = float(year)

    return feat


def get_matchup_features(
    home_team: str,
    away_team: str,
    final_features: List[str],
    year: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Build a feature vector for a matchup in the order of *final_features*.

    Returns None if data is unavailable for either team.
    """
    import sys

    def _log(msg: str) -> None:
        print(f"[serve_features] {msg}", file=sys.stderr)

    year = int(year or datetime.now().year)

    home_abb = TEAM_NAME_TO_ABB.get(home_team)
    away_abb = TEAM_NAME_TO_ABB.get(away_team)
    if home_abb is None:
        _log(f"unknown home team '{home_team}' — not in TEAM_NAME_TO_ABB")
        return None
    if away_abb is None:
        _log(f"unknown away team '{away_team}' — not in TEAM_NAME_TO_ABB")
        return None

    batting, pitching = _get_cached(year)
    if batting.empty:
        _log(f"batting DataFrame is empty for year={year} — run update_game_logs")
        return None
    if pitching.empty:
        _log(f"pitching DataFrame is empty for year={year} — run update_game_logs")
        return None

    home_bat = _latest(batting, home_abb)
    home_pit = _latest(pitching, home_abb)
    away_bat = _latest(batting, away_abb)
    away_pit = _latest(pitching, away_abb)

    missing = [
        label for label, val in [
            (f"home batting ({home_abb})", home_bat),
            (f"home pitching ({home_abb})", home_pit),
            (f"away batting ({away_abb})", away_bat),
            (f"away pitching ({away_abb})", away_pit),
        ] if val is None
    ]
    if missing:
        _log(f"no parquet rows found for: {', '.join(missing)}")
        return None

    feat_dict = _build_feat_dict(home_bat, home_pit, away_bat, away_pit, year)

    vec = np.array([feat_dict.get(f, np.nan) for f in final_features], dtype=float)
    nan_features = [f for f, v in zip(final_features, vec) if np.isnan(v)]
    if nan_features:
        _log(
            f"{len(nan_features)} NaN feature(s) for {home_team} vs {away_team}: "
            + ", ".join(nan_features[:10])
            + ("..." if len(nan_features) > 10 else "")
        )
        return None

    return vec


def invalidate_cache() -> None:
    """Force a reload of game log data on the next prediction call."""
    _cache["year"] = None
    _cache["batting"] = None
    _cache["pitching"] = None
