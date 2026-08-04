"""
SQLite store for predictions and resolved results.

Schema
------
predictions
  id            TEXT PK   (odds API game id)
  game_date     TEXT      (YYYY-MM-DD)
  home_team     TEXT
  away_team     TEXT
  predicted_winner TEXT
  home_win_prob REAL
  away_win_prob REAL
  is_value_pick INTEGER   (1 if any team had >=5% edge over implied)
  home_best_odds INTEGER  (American, best available)
  away_best_odds INTEGER
  model_used    TEXT
  created_at    TEXT
  actual_winner TEXT      (filled after game finishes)
  resolved      INTEGER   (0 = pending, 1 = resolved)
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "predictions.db")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as cx:
        cx.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              TEXT PRIMARY KEY,
                game_date       TEXT NOT NULL,
                home_team       TEXT NOT NULL,
                away_team       TEXT NOT NULL,
                predicted_winner TEXT,
                home_win_prob   REAL,
                away_win_prob   REAL,
                is_value_pick   INTEGER DEFAULT 0,
                home_best_odds  INTEGER,
                away_best_odds  INTEGER,
                model_used      TEXT,
                created_at      TEXT,
                actual_winner   TEXT,
                resolved        INTEGER DEFAULT 0
            )
        """)


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def upsert_prediction(game: dict, prediction: dict, value_alerts: list):
    best = game.get("best_odds", {})
    home_team = game["home_team"]
    away_team = game["away_team"]
    is_value = int(any(a["team"] in (home_team, away_team) for a in value_alerts))
    game_date = game["commence_time"][:10]

    with _conn() as cx:
        cx.execute("""
            INSERT INTO predictions
              (id, game_date, home_team, away_team, predicted_winner,
               home_win_prob, away_win_prob, is_value_pick,
               home_best_odds, away_best_odds, model_used, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO NOTHING
        """, (
            game["id"], game_date, home_team, away_team,
            prediction.get("predicted_winner"),
            prediction.get("home_win_prob"),
            prediction.get("away_win_prob"),
            is_value,
            best.get(home_team), best.get(away_team),
            prediction.get("model_used"),
            datetime.now(timezone.utc).isoformat(),
        ))


def get_unresolved(before_date: str) -> list[dict]:
    """Return unresolved predictions for dates before today."""
    with _conn() as cx:
        rows = cx.execute("""
            SELECT * FROM predictions
            WHERE resolved = 0 AND game_date < ?
        """, (before_date,)).fetchall()
    return [dict(r) for r in rows]


def resolve_game(game_id: str, actual_winner: str):
    with _conn() as cx:
        cx.execute("""
            UPDATE predictions
            SET actual_winner = ?, resolved = 1
            WHERE id = ?
        """, (actual_winner, game_id))


def get_performance_stats() -> dict:
    with _conn() as cx:
        rows = cx.execute("""
            SELECT * FROM predictions WHERE resolved = 1
        """).fetchall()

    rows = [dict(r) for r in rows]
    if not rows:
        return {"resolved_games": 0}

    def stats(subset):
        if not subset:
            return {"games": 0}
        correct = sum(1 for r in subset if r["predicted_winner"] == r["actual_winner"])
        roi = _calc_roi(subset)
        return {
            "games": len(subset),
            "correct": correct,
            "accuracy": round(correct / len(subset), 4),
            "roi": roi,
        }

    value_picks = [r for r in rows if r["is_value_pick"]]
    return {
        "resolved_games": len(rows),
        "overall": stats(rows),
        "value_picks": stats(value_picks),
        "by_model": _group_by(rows, "model_used", stats),
        "calibration": _calibration(rows),
    }


def _calc_roi(rows: list[dict]) -> float:
    """
    Simulated ROI betting $100 on predicted winner each game.
    Uses best available American odds for that team.
    """
    total_wagered = 0
    total_returned = 0
    for r in rows:
        winner = r["predicted_winner"]
        if winner == r["home_team"]:
            odds = r["home_best_odds"]
        else:
            odds = r["away_best_odds"]
        if odds is None:
            continue
        total_wagered += 100
        if r["predicted_winner"] == r["actual_winner"]:
            payout = (100 / abs(odds) * 100) if odds < 0 else (odds / 100 * 100)
            total_returned += 100 + payout
    if total_wagered == 0:
        return 0.0
    return round((total_returned - total_wagered) / total_wagered, 4)


def _group_by(rows, key, fn):
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {k: fn(v) for k, v in groups.items()}


def _calibration(rows: list[dict]) -> list[dict]:
    """Bucket predictions by model confidence and check actual win rate."""
    buckets: dict[str, list] = {}
    for r in rows:
        prob = r["home_win_prob"] if r["predicted_winner"] == r["home_team"] else r["away_win_prob"]
        if prob is None:
            continue
        bucket = f"{int(prob * 10) * 10}-{int(prob * 10) * 10 + 10}%"
        buckets.setdefault(bucket, []).append(r)
    result = []
    for bucket in sorted(buckets):
        subset = buckets[bucket]
        correct = sum(1 for r in subset if r["predicted_winner"] == r["actual_winner"])
        result.append({
            "bucket": bucket,
            "games": len(subset),
            "actual_win_rate": round(correct / len(subset), 4),
        })
    return result
