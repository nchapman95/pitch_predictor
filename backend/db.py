"""
SQLite store for predictions, resolved results, and full odds snapshots.

Schema
------
predictions
  id              TEXT PK   (odds API game id)
  game_date       TEXT      (YYYY-MM-DD)
  home_team       TEXT
  away_team       TEXT
  predicted_winner TEXT
  home_win_prob   REAL
  away_win_prob   REAL
  is_value_pick   INTEGER   (1 if any team had >=5% edge over implied)
  home_best_odds  INTEGER   (American, best available)
  away_best_odds  INTEGER
  odds_json       TEXT      (JSON: full per-book odds snapshot at prediction time)
  model_used      TEXT
  created_at      TEXT
  actual_winner   TEXT      (filled after game finishes)
  resolved        INTEGER   (0 = pending, 1 = resolved)

odds_snapshots
  id              INTEGER PK AUTOINCREMENT
  game_id         TEXT      (FK -> predictions.id)
  captured_at     TEXT      (ISO timestamp)
  odds_json       TEXT      (JSON: per-book lines at this moment)

  Allows tracking line movement by calling /api/games multiple times.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "predictions.db")

_PREDICTIONS_COLS = """
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
    odds_json       TEXT,
    model_used      TEXT,
    created_at      TEXT,
    actual_winner   TEXT,
    resolved        INTEGER DEFAULT 0
"""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as cx:
        cx.execute(f"CREATE TABLE IF NOT EXISTS predictions ({_PREDICTIONS_COLS})")
        cx.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id     TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                odds_json   TEXT NOT NULL
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_game ON odds_snapshots(game_id)")
        # Migrate: add odds_json column if upgrading from older schema
        _add_column_if_missing(cx, "predictions", "odds_json", "TEXT")


def _add_column_if_missing(cx, table, column, col_type):
    cols = [r[1] for r in cx.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        cx.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


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
    """
    Insert prediction if not already stored (first time we see this game_id).
    Always append a new odds snapshot so line movement is captured.
    """
    best = game.get("best_odds", {})
    odds = game.get("odds", {})
    home_team = game["home_team"]
    away_team = game["away_team"]
    is_value = int(any(a["team"] in (home_team, away_team) for a in value_alerts))
    game_date = game["commence_time"][:10]
    now = datetime.now(timezone.utc).isoformat()
    odds_json = json.dumps(odds)

    with _conn() as cx:
        cx.execute("""
            INSERT INTO predictions
              (id, game_date, home_team, away_team, predicted_winner,
               home_win_prob, away_win_prob, is_value_pick,
               home_best_odds, away_best_odds, odds_json, model_used, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO NOTHING
        """, (
            game["id"], game_date, home_team, away_team,
            prediction.get("predicted_winner"),
            prediction.get("home_win_prob"),
            prediction.get("away_win_prob"),
            is_value,
            best.get(home_team), best.get(away_team),
            odds_json,
            prediction.get("model_used"),
            now,
        ))

        # Always save an odds snapshot (captures line movement on repeated fetches)
        cx.execute("""
            INSERT INTO odds_snapshots (game_id, captured_at, odds_json)
            VALUES (?, ?, ?)
        """, (game["id"], now, odds_json))


def get_unresolved(before_date: str) -> list[dict]:
    with _conn() as cx:
        rows = cx.execute("""
            SELECT * FROM predictions
            WHERE resolved = 0 AND game_date < ?
        """, (before_date,)).fetchall()
    return [dict(r) for r in rows]


def resolve_game(game_id: str, actual_winner: str):
    with _conn() as cx:
        cx.execute("""
            UPDATE predictions SET actual_winner = ?, resolved = 1 WHERE id = ?
        """, (actual_winner, game_id))


def get_odds_history(game_id: str) -> list[dict]:
    """Return all odds snapshots for a game, oldest first."""
    with _conn() as cx:
        rows = cx.execute("""
            SELECT captured_at, odds_json FROM odds_snapshots
            WHERE game_id = ? ORDER BY id ASC
        """, (game_id,)).fetchall()
    return [{"captured_at": r["captured_at"], "odds": json.loads(r["odds_json"])} for r in rows]


def get_performance_stats() -> dict:
    with _conn() as cx:
        rows = cx.execute("SELECT * FROM predictions WHERE resolved = 1").fetchall()
    rows = [dict(r) for r in rows]
    if not rows:
        return {"resolved_games": 0}

    def stats(subset):
        if not subset:
            return {"games": 0}
        correct = sum(1 for r in subset if r["predicted_winner"] == r["actual_winner"])
        return {
            "games": len(subset),
            "correct": correct,
            "accuracy": round(correct / len(subset), 4),
            "roi": _calc_roi(subset),
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
    total_wagered = total_returned = 0
    for r in rows:
        winner = r["predicted_winner"]
        odds = r["home_best_odds"] if winner == r["home_team"] else r["away_best_odds"]
        if odds is None:
            continue
        total_wagered += 100
        if r["predicted_winner"] == r["actual_winner"]:
            payout = (100 / abs(odds) * 100) if odds < 0 else (odds / 100 * 100)
            total_returned += 100 + payout
    return 0.0 if total_wagered == 0 else round((total_returned - total_wagered) / total_wagered, 4)


def _group_by(rows, key, fn):
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return {k: fn(v) for k, v in groups.items()}


def _calibration(rows: list[dict]) -> list[dict]:
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
