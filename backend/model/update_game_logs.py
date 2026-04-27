"""
Refresh game log parquets for all MLB teams.

Usage:
    python -m model.update_game_logs [--years 2024 2025] [--force]

Default: refreshes only the current year.
Historical seasons are skipped when all team parquets already exist,
unless --force is supplied.
"""

import argparse
import os
import time
import warnings
from datetime import datetime
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

_REQUEST_DELAY = 3.5  # seconds between BRef requests

try:
    from pybaseball.datasources.bref import BRefSession
    _session = BRefSession()
except Exception as exc:
    raise ImportError("pybaseball is required for update_game_logs") from exc

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "game_log_data")
_URL = "https://www.baseball-reference.com/teams/tgl.cgi?team={}&t={}&year={}"

# Canonical roster — OAK becomes ATH on BRef starting 2025
_TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]
_ATH_YEAR = 2025  # BRef switched OAK → ATH from this season on


def _bref_abb(team: str, year: int) -> str:
    return "ATH" if team == "OAK" and year >= _ATH_YEAR else team


def _parquet_path(log_type: str, team: str, year: int) -> str:
    file_abb = _bref_abb(team, year)
    return os.path.join(ARTIFACTS_DIR, f"{log_type}_{file_abb}_{year}_logs.parquet")


def _season_complete(year: int) -> bool:
    """Heuristic: treat prior seasons as complete."""
    return year < datetime.now().year


def _all_files_exist(year: int) -> bool:
    for team in _TEAMS:
        for log_type in ("batting", "pitching"):
            if not os.path.exists(_parquet_path(log_type, team, year)):
                return False
    return True


def _fetch_raw(year: int, team: str, log_type: str) -> pd.DataFrame:
    bref_abb = _bref_abb(team, year)
    t_param = "b" if log_type == "batting" else "p"
    url = _URL.format(bref_abb, t_param, year)
    content = _session.get(url).content
    soup = BeautifulSoup(content, "html.parser")
    table_id = f"players_standard_{log_type}"
    table = soup.find("table", attrs={"id": table_id})
    if table is None:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = pd.read_html(StringIO(str(table)))[0]
    return data


def _postprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Clean up the multi-level BRef table into a flat, typed DataFrame."""
    if not isinstance(data.columns, pd.MultiIndex):
        return pd.DataFrame()

    # Drop rank column and totals row
    rk_cols = [c for c in data.columns if "Rk" in str(c)]
    data = data.drop(columns=rk_cols, errors="ignore")
    data = data.iloc[:-1].copy()

    # Rename columns at any level
    repl = {
        "Gtm": "Game",
        "Unnamed: 3_level_1": "Home",
        "#": "NumPlayers",
        "Opp. Starter (GmeSc)": "OppStart",
        "Pitchers Used (Rest-GameScore-Dec)": "PitchersUsed",
    }
    data = data.rename(columns=repl)

    # Home/away indicator: '@' = away, blank = home
    home_key = None
    for col in data.columns:
        if isinstance(col, tuple) and "Home" in col:
            home_key = col
            break
    if home_key:
        data[home_key] = data[home_key].isnull()

    # Drop repeated header rows (BRef inserts month sub-headers)
    game_key = None
    for col in data.columns:
        if isinstance(col, tuple) and "Game" in col:
            game_key = col
            break
    if game_key:
        data = data[data[game_key] != "Gtm"].copy()

    # Convert to numeric where possible
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = data.apply(pd.to_numeric, errors="ignore")

    if game_key and game_key in data.columns:
        data[game_key] = pd.to_numeric(data[game_key], errors="coerce")
        data = data.dropna(subset=[game_key]).copy()
        data[game_key] = data[game_key].astype(int)

    # Flatten to single-level columns
    data.columns = data.columns.droplevel(0)

    return data.reset_index(drop=True)


def fetch_and_save(year: int, team: str, log_type: str) -> bool:
    """Fetch one team-season log and save to parquet. Returns True on success."""
    try:
        time.sleep(_REQUEST_DELAY)
        raw = _fetch_raw(year, team, log_type)
        if raw.empty:
            return False
        df = _postprocess(raw)
        if df.empty:
            return False
        path = _parquet_path(log_type, team, year)
        df.to_parquet(path, index=False)
        return True
    except Exception as exc:
        print(f"    ERROR {team} {year} {log_type}: {exc}")
        return False


def update(years: list, force: bool = False) -> None:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    for year in years:
        if not force and _season_complete(year) and _all_files_exist(year):
            print(f"{year}: all files exist, skipping (use --force to refresh)")
            continue

        print(f"\n{year}: fetching logs...")
        ok = fail = 0
        for team in _TEAMS:
            for log_type in ("batting", "pitching"):
                path = _parquet_path(log_type, team, year)
                if not force and os.path.exists(path) and _season_complete(year):
                    ok += 1
                    continue
                success = fetch_and_save(year, team, log_type)
                if success:
                    ok += 1
                else:
                    fail += 1

        print(f"{year}: {ok} saved, {fail} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=[datetime.now().year],
        help="Seasons to update (default: current year only)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch even if parquet already exists",
    )
    args = parser.parse_args()
    update(years=args.years, force=args.force)
