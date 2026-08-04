"""
Backfill model predictions for games already stored in the database.
Odds are NOT re-fetched — this script is purely about running (or re-running)
the ML models against games we already have.

Use cases:
  • After training model_v2.pkl: populate model_predictions for all historical games
  • After training a new model: add its rows without touching odds data
  • Re-run a specific model with updated weights (--force)

Usage:
    # Backfill all missing model predictions for 2026
    python backfill_predictions.py --start 2026-01-01 --end 2026-12-31

    # Only v2 pitcher model
    python backfill_predictions.py --start 2026-01-01 --end 2026-12-31 --models v2

    # Force overwrite even if predictions already exist
    python backfill_predictions.py --start 2026-01-01 --end 2026-12-31 --force

    # Dry run: show what would be processed
    python backfill_predictions.py --start 2026-01-01 --end 2026-12-31 --dry-run
"""

import argparse
import os
import sqlite3
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
import db
from model.predictor import MultiPredictor

SLEEP_BETWEEN_DATES = 0.2   # seconds — be polite to the MLB Stats API


# ---------------------------------------------------------------------------
# DB helpers (read-only queries not in db.py)
# ---------------------------------------------------------------------------

def _get_games(start: str, end: str) -> list[dict]:
    """Return all game records in the date range from the predictions table."""
    con = sqlite3.connect(db.DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT id, game_date, home_team, away_team,
               home_best_odds, away_best_odds
        FROM predictions
        WHERE game_date >= ? AND game_date <= ?
        ORDER BY game_date, id
    """, (start, end)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def _already_predicted(game_id: str, model_name: str) -> bool:
    con = sqlite3.connect(db.DB_PATH)
    row = con.execute(
        "SELECT 1 FROM model_predictions WHERE game_id = ? AND model_name = ?",
        (game_id, model_name)
    ).fetchone()
    con.close()
    return row is not None


def _delete_model_predictions(game_id: str, model_name: str):
    con = sqlite3.connect(db.DB_PATH)
    con.execute(
        "DELETE FROM model_predictions WHERE game_id = ? AND model_name = ?",
        (game_id, model_name)
    )
    con.commit()
    con.close()


# ---------------------------------------------------------------------------
# Value alert helper
# ---------------------------------------------------------------------------

def _implied(odds):
    if odds is None:
        return None
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)


def _is_value(home_team, away_team, home_best_odds, away_best_odds, prediction) -> int:
    for team, odds, prob_key in [
        (home_team, home_best_odds, "home_win_prob"),
        (away_team, away_best_odds, "away_win_prob"),
    ]:
        imp = _implied(odds)
        if imp is None:
            continue
        if prediction.get(prob_key, 0) - imp >= 0.05:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Main backfill logic
# ---------------------------------------------------------------------------

def run(start: str, end: str, models_filter: list, force: bool, dry_run: bool):
    db.init_db()   # ensures model_predictions table exists

    predictor = MultiPredictor()
    loaded    = predictor.models()
    if not loaded:
        print("No models found in artifacts/. Train a model first.")
        sys.exit(1)

    target_models = [m for m in loaded if not models_filter or m in models_filter]
    if not target_models:
        print(f"No matching models. Loaded: {loaded}  Requested: {models_filter}")
        sys.exit(1)

    print(f"Models to backfill : {target_models}")
    print(f"Date range         : {start} → {end}")
    print(f"Force overwrite    : {force}")
    print(f"Dry run            : {dry_run}")

    games = _get_games(start, end)
    if not games:
        print("No games found in the database for that date range.")
        return

    print(f"Games in DB        : {len(games)}")
    print()

    # Group games by date so we only fetch starters once per date
    by_date: dict[str, list] = defaultdict(list)
    for g in games:
        by_date[g["game_date"]].append(g)

    total_written = defaultdict(int)
    total_skipped = defaultdict(int)

    for date_str, day_games in sorted(by_date.items()):
        # Fetch probable/actual starters for the date (free MLB Stats API)
        starters = {}
        try:
            from model.pitcher_features import get_starters_for_date
            starters = get_starters_for_date(date_str)
        except Exception as exc:
            print(f"  [warn] starter fetch failed for {date_str}: {exc}")

        print(f"{date_str}  ({len(day_games)} games)", end="")

        date_written = 0
        for game in day_games:
            game_id   = game["id"]
            home_team = game["home_team"]
            away_team = game["away_team"]

            # Match probable starters for this matchup
            home_pitcher = away_pitcher = None
            for matchup_data in starters.values():
                h = matchup_data.get("home_team", "")
                a = matchup_data.get("away_team", "")
                if (h[:3].upper() == home_team[:3].upper() or
                        a[:3].upper() == away_team[:3].upper()):
                    home_pitcher = matchup_data.get("home_pitcher")
                    away_pitcher = matchup_data.get("away_pitcher")
                    break

            all_preds = predictor.predict_all(
                home_team, away_team,
                home_pitcher=home_pitcher,
                away_pitcher=away_pitcher,
            )

            for model_name in target_models:
                pred = all_preds.get(model_name)
                if pred is None:
                    continue

                already = _already_predicted(game_id, model_name)
                if already and not force:
                    total_skipped[model_name] += 1
                    continue

                if dry_run:
                    total_written[model_name] += 1
                    continue

                if already and force:
                    _delete_model_predictions(game_id, model_name)

                is_val = _is_value(
                    home_team, away_team,
                    game.get("home_best_odds"), game.get("away_best_odds"),
                    pred,
                )
                db.upsert_model_prediction(game_id, model_name, pred, is_val)
                total_written[model_name] += 1
                date_written += 1

        print(f"  +{date_written} written")
        time.sleep(SLEEP_BETWEEN_DATES)

    print()
    for model in target_models:
        action = "would write" if dry_run else "wrote"
        print(f"  {model}: {action} {total_written[model]}, "
              f"skipped {total_skipped[model]}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill model predictions for games already in the DB"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--models", nargs="+", default=[],
        help="Which models to run (e.g. --models v1 v2). Default: all loaded models."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing model_predictions rows (default: skip if already stored)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be written without touching the DB"
    )
    args = parser.parse_args()
    run(
        start=args.start,
        end=args.end,
        models_filter=args.models,
        force=args.force,
        dry_run=args.dry_run,
    )
