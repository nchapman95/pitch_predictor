import os
import sys
from fastapi import FastAPI, HTTPException, Query
from datetime import date as Date, datetime, timezone
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from odds_client import OddsClient
from model.predictor import MultiPredictor
import db
import results as results_module

load_dotenv()

app = FastAPI(title="MLB Game Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

odds_client = OddsClient(api_key=os.getenv("ODDS_API_KEY", ""))
predictor   = MultiPredictor()
db.init_db()

# In-memory starter cache: date_str → {matchup_key: {home_pitcher, away_pitcher, ...}}
_starter_cache: dict = {}


def _get_starters(date_str: str) -> dict:
    if date_str not in _starter_cache:
        try:
            from model.pitcher_features import get_starters_for_date
            _starter_cache[date_str] = get_starters_for_date(date_str)
        except Exception as exc:
            print(f"[main] starter fetch failed for {date_str}: {exc}", file=sys.stderr)
            _starter_cache[date_str] = {}
    return _starter_cache[date_str]


def _find_pitchers(starters: dict, home_team: str, away_team: str):
    for matchup_data in starters.values():
        h = matchup_data.get("home_team", "")
        a = matchup_data.get("away_team", "")
        if (h[:3].upper() == home_team[:3].upper() or
                a[:3].upper() == away_team[:3].upper()):
            return (
                matchup_data.get("home_pitcher"),
                matchup_data.get("away_pitcher"),
                matchup_data.get("home_pitcher_id"),
                matchup_data.get("away_pitcher_id"),
            )
    return None, None, None, None


def _value_alerts(home_team, away_team, best_odds, prediction):
    def implied(odds):
        if odds is None:
            return None
        return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

    alerts = []
    for team, prob_key in [(home_team, "home_win_prob"), (away_team, "away_win_prob")]:
        odds = best_odds.get(team)
        imp  = implied(odds)
        if imp is None:
            continue
        edge = prediction.get(prob_key, 0) - imp
        if edge >= 0.05:
            alerts.append({"team": team, "edge": round(edge, 4)})
    return alerts


@app.get("/api/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": predictor.is_loaded(),
        "models":       predictor.models(),
    }


@app.get("/api/games")
def get_games(date: Date = Query(default=None)):
    """Return MLB games with live odds, probable starters, and per-model ML predictions."""
    try:
        games = odds_client.get_todays_mlb_games(date=date)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Odds API error: {e}")

    date_str = (date or datetime.now(tz=_ET).date()).isoformat()
    starters = _get_starters(date_str)

    response = []
    for game in games:
        home_team = game["home_team"]
        away_team = game["away_team"]
        best_odds = game.get("best_odds", {})

        home_pitcher, away_pitcher, home_pid, away_pid = _find_pitchers(
            starters, home_team, away_team
        )

        all_predictions = predictor.predict_all(
            home_team, away_team,
            home_pitcher=home_pitcher,
            away_pitcher=away_pitcher,
        )

        # Store each model's prediction separately
        for model_name, pred in all_predictions.items():
            alerts   = _value_alerts(home_team, away_team, best_odds, pred)
            is_value = int(bool(alerts))
            db.upsert_model_prediction(game["id"], model_name, pred, is_value)

        # Game record (uses primary / best model for backward-compat columns)
        primary_pred   = predictor.primary_prediction(all_predictions)
        primary_alerts = _value_alerts(home_team, away_team, best_odds, primary_pred)
        db.upsert_prediction(game, primary_pred, primary_alerts)

        response.append({
            **game,
            "home_pitcher":    home_pitcher,
            "away_pitcher":    away_pitcher,
            "home_pitcher_id": home_pid,
            "away_pitcher_id": away_pid,
            "predictions":     all_predictions,
        })

    return {"games": response}


@app.get("/api/performance")
def get_performance():
    """Resolve completed games then return per-model prediction stats."""
    newly_resolved = results_module.resolve_pending(db)
    stats = db.get_performance_stats()
    return {**stats, "newly_resolved": newly_resolved}


@app.get("/api/odds/raw")
def get_raw_odds():
    """Raw odds response from The Odds API (for debugging)."""
    return odds_client.get_raw()
