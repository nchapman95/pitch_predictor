"""
Fetch final MLB game scores from the free MLB Stats API and resolve predictions.
No API key required.

Usage (CLI):
    python results.py                          # resolve all pending past games
    python results.py --start 2026-08-01       # resolve from a specific date
    python results.py --start 2026-08-01 --end 2026-08-31
"""

import argparse
import requests
import sys
import time
from datetime import datetime, date, timedelta, timezone

MLB_API = "https://statsapi.mlb.com/api/v1"

# MLB Stats API team name -> Odds API team name (handle common mismatches)
_NAME_MAP = {
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
    "Athletics": "Athletics",
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
    "Athletics": "Oakland Athletics",
}


def get_final_results(date_str: str) -> list[dict]:
    """
    Return list of {home_team, away_team, winner} for final games on date_str (YYYY-MM-DD).
    Uses Odds API team names.
    """
    url = f"{MLB_API}/schedule"
    params = {"sportId": 1, "date": date_str, "hydrate": "linescore"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception:
        return []

    results = []
    for date_entry in resp.json().get("dates", []):
        for game in date_entry.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_name = _NAME_MAP.get(home.get("team", {}).get("name", ""))
            away_name = _NAME_MAP.get(away.get("team", {}).get("name", ""))
            if not home_name or not away_name:
                continue
            winner = home_name if home.get("isWinner") else away_name
            results.append({
                "home_team": home_name,
                "away_team": away_name,
                "winner": winner,
            })
    return results


def resolve_pending(db):
    """
    Pull unresolved predictions for past dates and fill in actual_winner.
    Call this on startup and when /api/performance is requested.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    unresolved = db.get_unresolved(before_date=today)
    if not unresolved:
        return 0

    by_date: dict[str, list] = {}
    for row in unresolved:
        by_date.setdefault(row["game_date"], []).append(row)

    resolved_count = 0
    for date_str, rows in by_date.items():
        finals = get_final_results(date_str)
        for row in rows:
            match = next(
                (f for f in finals
                 if f["home_team"] == row["home_team"] and f["away_team"] == row["away_team"]),
                None,
            )
            if match:
                db.resolve_game(row["id"], match["winner"])
                resolved_count += 1

    return resolved_count


def resolve_range(db, start: str, end: str, debug: bool = False) -> int:
    """
    Resolve all games (resolved or not) in a date range.
    Useful for fixing missed results or re-checking a specific month.
    """
    con_rows = db.get_unresolved(before_date="9999-12-31")   # all unresolved ever
    in_range = [r for r in con_rows if start <= r["game_date"] <= end]

    by_date: dict[str, list] = {}
    for row in in_range:
        by_date.setdefault(row["game_date"], []).append(row)

    resolved_count = 0
    dates = sorted(by_date)
    for i, date_str in enumerate(dates):
        finals = get_final_results(date_str)
        rows   = by_date[date_str]
        day_resolved = 0
        for row in rows:
            match = next(
                (f for f in finals
                 if f["home_team"] == row["home_team"] and f["away_team"] == row["away_team"]),
                None,
            )
            if match:
                db.resolve_game(row["id"], match["winner"])
                resolved_count += 1
                day_resolved += 1
            elif debug:
                print(f"    NO MATCH for: {row['away_team']} @ {row['home_team']}")
                print(f"    API returned: {[(f['away_team'], f['home_team']) for f in finals]}")
        print(f"  {date_str}  {day_resolved}/{len(rows)} resolved")
        if i < len(dates) - 1:
            time.sleep(0.2)

    return resolved_count


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    import db

    parser = argparse.ArgumentParser(description="Resolve MLB game results from the MLB Stats API")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (default: all pending)")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--debug", action="store_true", help="Print unmatched games and API names")
    args = parser.parse_args()

    db.init_db()

    if args.start:
        end = args.end or (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        print(f"Resolving {args.start} → {end} ...")
        n = resolve_range(db, args.start, end, debug=args.debug)
    else:
        print("Resolving all pending past games ...")
        n = resolve_pending(db)

    print(f"\nDone — {n} game(s) resolved.")
