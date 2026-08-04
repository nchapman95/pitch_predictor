"""
One-time historical odds backfill.

Fetches a pre-game odds snapshot (noon UTC) for each date in a range
from The Odds API historical endpoint (requires paid plan).

Credit cost: ~1 credit per event returned (~10-16 per MLB game day).
With 20,000 credits you can cover roughly 7 full seasons.

Usage:
    # Preview credit estimate without fetching anything
    python backfill_odds.py --start 2024-04-01 --end 2024-09-30 --dry-run

    # Run the actual backfill
    python backfill_odds.py --start 2024-04-01 --end 2024-09-30

    # Specify a different key (otherwise reads from .env)
    python backfill_odds.py --start 2024-04-01 --end 2024-09-30 --api-key YOUR_KEY

Dates already in backfill_log are automatically skipped — safe to re-run.
"""

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta

import requests
from dotenv import load_dotenv

# Allow running from the backend/ dir directly
sys.path.insert(0, os.path.dirname(__file__))
import db
from odds_client import ODDS_API_BASE, SPORT, BOOKMAKERS, _parse_bookmaker_odds, _best_odds

load_dotenv()

SNAPSHOT_TIME = "T12:00:00Z"   # noon UTC — pre-game for all MLB start times
SLEEP_BETWEEN = 1.0            # seconds between requests (be a good API citizen)


# ---------------------------------------------------------------------------
# Backfill log table
# ---------------------------------------------------------------------------

def _ensure_backfill_log():
    import sqlite3
    from contextlib import contextmanager

    @contextmanager
    def conn():
        con = sqlite3.connect(db.DB_PATH)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    with conn() as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS backfill_log (
                date              TEXT PRIMARY KEY,
                fetched_at        TEXT,
                events_fetched    INTEGER,
                credits_used      INTEGER,
                credits_remaining INTEGER
            )
        """)


def _already_fetched(date_str: str) -> bool:
    import sqlite3
    con = sqlite3.connect(db.DB_PATH)
    row = con.execute(
        "SELECT 1 FROM backfill_log WHERE date = ?", (date_str,)
    ).fetchone()
    con.close()
    return row is not None


def _log_fetch(date_str, events, credits_used, credits_remaining):
    import sqlite3
    from datetime import datetime, timezone
    con = sqlite3.connect(db.DB_PATH)
    con.execute("""
        INSERT OR REPLACE INTO backfill_log
          (date, fetched_at, events_fetched, credits_used, credits_remaining)
        VALUES (?, ?, ?, ?, ?)
    """, (date_str, datetime.now(timezone.utc).isoformat(), events, credits_used, credits_remaining))
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Fetch + save
# ---------------------------------------------------------------------------

def fetch_historical(date_str: str, api_key: str) -> tuple[list, int, int]:
    """
    Returns (events, credits_used, credits_remaining).
    credits_used/remaining come from response headers.
    """
    url = f"{ODDS_API_BASE}/historical/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "date": date_str + SNAPSHOT_TIME,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    credits_used = int(resp.headers.get("x-requests-used", 0))
    credits_remaining = int(resp.headers.get("x-requests-remaining", 0))

    data = resp.json()
    events = data.get("data", []) if isinstance(data, dict) else data
    return events, credits_used, credits_remaining


def save_events(events: list, date_str: str, api_key_hint: str):
    """Parse events and save odds + baseline predictions to the DB."""
    from model.predictor import GamePredictor
    predictor = GamePredictor()

    for event in events:
        # Only save events that actually fall on this date
        game_date = event.get("commence_time", "")[:10]
        if game_date != date_str:
            continue

        odds_by_book = _parse_bookmaker_odds(event.get("bookmakers", []))
        best = _best_odds(odds_by_book)

        game = {
            "id": event["id"],
            "home_team": event["home_team"],
            "away_team": event["away_team"],
            "commence_time": event["commence_time"],
            "odds": odds_by_book,
            "best_odds": best,
        }
        prediction = predictor.predict(event["home_team"], event["away_team"])

        # Compute value alerts inline
        alerts = _value_alerts(event["home_team"], event["away_team"], best, prediction)
        db.upsert_prediction(game, prediction, alerts)


def _value_alerts(home_team, away_team, best_odds, prediction):
    alerts = []
    for team, prob_key in [(home_team, "home_win_prob"), (away_team, "away_win_prob")]:
        odds = best_odds.get(team)
        if odds is None:
            continue
        implied = abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)
        edge = prediction.get(prob_key, 0) - implied
        if edge >= 0.05:
            alerts.append({"team": team, "edge": edge})
    return alerts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def run(start: date, end: date, api_key: str, dry_run: bool):
    db.init_db()
    _ensure_backfill_log()

    all_dates = list(daterange(start, end))
    already_done = [d for d in all_dates if _already_fetched(d.isoformat())]
    to_fetch = [d for d in all_dates if not _already_fetched(d.isoformat())]

    print(f"\nDate range : {start} → {end}  ({len(all_dates)} days)")
    print(f"Already fetched : {len(already_done)} days (skipping)")
    print(f"To fetch        : {len(to_fetch)} days")

    # MLB regular season is ~180 days; estimate ~13 games/day on average
    est_credits = len(to_fetch) * 13
    print(f"Credit estimate : ~{est_credits} credits ({len(to_fetch)} days × ~13 events/day)")

    if dry_run:
        print("\n[dry-run] No requests made.")
        return

    if not to_fetch:
        print("\nNothing to fetch.")
        return

    print()
    total_credits_used = 0

    for i, d in enumerate(to_fetch):
        date_str = d.isoformat()
        try:
            events, used, remaining = fetch_historical(date_str, api_key)
            save_events(events, date_str, api_key)
            _log_fetch(date_str, len(events), used, remaining)
            total_credits_used += used
            print(f"[{i+1}/{len(to_fetch)}] {date_str}  {len(events)} events  "
                  f"used={used}  remaining={remaining}")
        except requests.HTTPError as e:
            if e.response.status_code in (401, 402, 403):
                print(f"\n✗ API key rejected on {date_str} (status {e.response.status_code}). "
                      f"Historical endpoint requires a paid plan.")
                sys.exit(1)
            print(f"[{i+1}/{len(to_fetch)}] {date_str}  ERROR: {e}")
        except Exception as e:
            print(f"[{i+1}/{len(to_fetch)}] {date_str}  ERROR: {e}")

        if i < len(to_fetch) - 1:
            time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. Total credits used this run: {total_credits_used}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical MLB odds")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--api-key", default=None, help="Odds API key (default: ODDS_API_KEY env var)")
    parser.add_argument("--dry-run", action="store_true", help="Estimate credits without fetching")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ODDS_API_KEY", "")
    if not api_key:
        print("Error: set ODDS_API_KEY in .env or pass --api-key")
        sys.exit(1)

    run(
        start=date.fromisoformat(args.start),
        end=date.fromisoformat(args.end),
        api_key=api_key,
        dry_run=args.dry_run,
    )
