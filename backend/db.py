"""
SQLite store for predictions, resolved results, and full odds snapshots.

Schema
------
predictions          — one row per game (game record + odds + resolved result)
model_predictions    — one row per (game, model); stores each model's pick
odds_snapshots       — one row per /api/games call; captures line movement
backfill_log         — tracks which dates have been backfilled
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
    home_implied_prob REAL,
    away_implied_prob REAL,
    odds_json       TEXT,
    model_used      TEXT,
    created_at      TEXT,
    actual_winner   TEXT,
    resolved        INTEGER DEFAULT 0
"""

_MODEL_PREDICTIONS_COLS = """
    game_id         TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    predicted_winner TEXT,
    home_win_prob   REAL,
    away_win_prob   REAL,
    is_value_pick   INTEGER DEFAULT 0,
    created_at      TEXT,
    PRIMARY KEY (game_id, model_name)
"""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _conn() as cx:
        cx.execute(f"CREATE TABLE IF NOT EXISTS predictions ({_PREDICTIONS_COLS})")
        cx.execute(f"CREATE TABLE IF NOT EXISTS model_predictions ({_MODEL_PREDICTIONS_COLS})")
        cx.execute("""
            CREATE TABLE IF NOT EXISTS odds_snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id     TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                odds_json   TEXT NOT NULL
            )
        """)
        cx.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_game ON odds_snapshots(game_id)")
        cx.execute("CREATE INDEX IF NOT EXISTS idx_mp_game ON model_predictions(game_id)")
        _add_column_if_missing(cx, "predictions", "odds_json", "TEXT")
        _add_column_if_missing(cx, "predictions", "home_implied_prob", "REAL")
        _add_column_if_missing(cx, "predictions", "away_implied_prob", "REAL")
        _backfill_implied_probs(cx)


def _american_to_implied(odds):
    """Convert American moneyline odds to implied win probability."""
    if odds is None:
        return None
    if odds < 0:
        return round(abs(odds) / (abs(odds) + 100), 6)
    return round(100 / (odds + 100), 6)


def _add_column_if_missing(cx, table, column, col_type):
    cols = [r[1] for r in cx.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        cx.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def _backfill_implied_probs(cx):
    """One-time migration: populate implied probs for rows that have odds but no prob yet."""
    rows = cx.execute("""
        SELECT id, home_best_odds, away_best_odds
        FROM predictions
        WHERE home_implied_prob IS NULL
          AND (home_best_odds IS NOT NULL OR away_best_odds IS NOT NULL)
    """).fetchall()
    for row in rows:
        cx.execute("""
            UPDATE predictions
            SET home_implied_prob = ?, away_implied_prob = ?
            WHERE id = ?
        """, (
            _american_to_implied(row[1]),
            _american_to_implied(row[2]),
            row[0],
        ))


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
    Insert game record + primary model prediction (first write wins).
    Always appends an odds snapshot for line-movement tracking.
    """
    best      = game.get("best_odds", {})
    odds      = game.get("odds", {})
    home_team = game["home_team"]
    away_team = game["away_team"]
    is_value  = int(any(a["team"] in (home_team, away_team) for a in value_alerts))
    game_date = game["commence_time"][:10]
    now       = datetime.now(timezone.utc).isoformat()
    odds_json = json.dumps(odds)

    home_odds = best.get(home_team)
    away_odds = best.get(away_team)

    with _conn() as cx:
        cx.execute("""
            INSERT INTO predictions
              (id, game_date, home_team, away_team, predicted_winner,
               home_win_prob, away_win_prob, is_value_pick,
               home_best_odds, away_best_odds,
               home_implied_prob, away_implied_prob,
               odds_json, model_used, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO NOTHING
        """, (
            game["id"], game_date, home_team, away_team,
            prediction.get("predicted_winner"),
            prediction.get("home_win_prob"),
            prediction.get("away_win_prob"),
            is_value,
            home_odds, away_odds,
            _american_to_implied(home_odds),
            _american_to_implied(away_odds),
            odds_json,
            prediction.get("model_used"),
            now,
        ))

        cx.execute("""
            INSERT INTO odds_snapshots (game_id, captured_at, odds_json)
            VALUES (?, ?, ?)
        """, (game["id"], now, odds_json))


def upsert_model_prediction(game_id: str, model_name: str,
                            prediction: dict, is_value: int):
    """
    Store one model's prediction for a game (first write wins per model).
    """
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as cx:
        cx.execute("""
            INSERT INTO model_predictions
              (game_id, model_name, predicted_winner,
               home_win_prob, away_win_prob, is_value_pick, created_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(game_id, model_name) DO NOTHING
        """, (
            game_id, model_name,
            prediction.get("predicted_winner"),
            prediction.get("home_win_prob"),
            prediction.get("away_win_prob"),
            is_value,
            now,
        ))


def get_unresolved(before_date: str) -> list:
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


def get_odds_history(game_id: str) -> list:
    with _conn() as cx:
        rows = cx.execute("""
            SELECT captured_at, odds_json FROM odds_snapshots
            WHERE game_id = ? ORDER BY id ASC
        """, (game_id,)).fetchall()
    return [{"captured_at": r["captured_at"], "odds": json.loads(r["odds_json"])} for r in rows]


def get_performance_stats() -> dict:
    """
    Returns per-model accuracy / ROI stats by joining model_predictions
    with the resolved game records in predictions.
    Falls back to the predictions table's own model_used column for data
    written before model_predictions existed.
    """
    with _conn() as cx:
        # Per-model rows from model_predictions (preferred)
        mp_rows = cx.execute("""
            SELECT mp.game_id, mp.model_name,
                   mp.predicted_winner, mp.home_win_prob, mp.away_win_prob,
                   mp.is_value_pick,
                   p.actual_winner, p.home_team, p.away_team,
                   p.home_best_odds, p.away_best_odds
            FROM model_predictions mp
            JOIN predictions p ON mp.game_id = p.id
            WHERE p.resolved = 1
        """).fetchall()

        # Legacy rows stored only in predictions (no model_predictions entry)
        legacy_rows = cx.execute("""
            SELECT p.id AS game_id, p.model_used AS model_name,
                   p.predicted_winner, p.home_win_prob, p.away_win_prob,
                   p.is_value_pick,
                   p.actual_winner, p.home_team, p.away_team,
                   p.home_best_odds, p.away_best_odds
            FROM predictions p
            WHERE p.resolved = 1
              AND NOT EXISTS (
                  SELECT 1 FROM model_predictions mp WHERE mp.game_id = p.id
              )
        """).fetchall()

    all_rows = [dict(r) for r in mp_rows] + [dict(r) for r in legacy_rows]

    if not all_rows:
        return {"resolved_games": 0, "by_model": {}}

    # Count unique resolved games
    resolved_games = len({r["game_id"] for r in all_rows})

    # Group by model
    by_model_rows: dict[str, list] = {}
    for r in all_rows:
        by_model_rows.setdefault(r["model_name"] or "unknown", []).append(r)

    by_model = {
        model: _model_stats(rows)
        for model, rows in by_model_rows.items()
    }

    return {
        "resolved_games": resolved_games,
        "by_model": by_model,
    }


def _model_stats(rows: list) -> dict:
    def stats(subset):
        if not subset:
            return {"games": 0}
        correct = sum(1 for r in subset if r["predicted_winner"] == r["actual_winner"])
        return {
            "games":    len(subset),
            "correct":  correct,
            "accuracy": round(correct / len(subset), 4),
            "roi":      _calc_roi(subset),
        }

    value_picks = [r for r in rows if r.get("is_value_pick")]
    return {
        "overall":     stats(rows),
        "value_picks": stats(value_picks),
        "calibration": _calibration(rows),
    }


def _calc_roi(rows: list) -> float:
    total_wagered = total_returned = 0
    for r in rows:
        winner = r["predicted_winner"]
        odds   = r["home_best_odds"] if winner == r["home_team"] else r["away_best_odds"]
        if odds is None:
            continue
        total_wagered += 100
        if r["predicted_winner"] == r["actual_winner"]:
            payout = (100 / abs(odds) * 100) if odds < 0 else (odds / 100 * 100)
            total_returned += 100 + payout
    return 0.0 if total_wagered == 0 else round(
        (total_returned - total_wagered) / total_wagered, 4
    )


def _calibration(rows: list) -> list:
    buckets: dict[str, list] = {}
    for r in rows:
        prob = r["home_win_prob"] if r["predicted_winner"] == r["home_team"] else r["away_win_prob"]
        if prob is None:
            continue
        bucket = f"{int(prob * 10) * 10}-{int(prob * 10) * 10 + 10}%"
        buckets.setdefault(bucket, []).append(r)
    result = []
    for bucket in sorted(buckets):
        subset  = buckets[bucket]
        correct = sum(1 for r in subset if r["predicted_winner"] == r["actual_winner"])
        result.append({
            "bucket":          bucket,
            "games":           len(subset),
            "actual_win_rate": round(correct / len(subset), 4),
        })
    return result
