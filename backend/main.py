import os
from fastapi import FastAPI, HTTPException, Query
from datetime import date as Date, datetime, timezone
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from odds_client import OddsClient
from model.predictor import GamePredictor
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
predictor = GamePredictor()
db.init_db()


def _value_alerts(home_team, away_team, best_odds, prediction):
    """Mirror of frontend logic — needed to store is_value_pick flag."""
    def implied(odds):
        if odds is None:
            return None
        return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

    alerts = []
    for team, prob_key in [(home_team, "home_win_prob"), (away_team, "away_win_prob")]:
        odds = best_odds.get(team)
        imp = implied(odds)
        if imp is None:
            continue
        edge = prediction.get(prob_key, 0) - imp
        if edge >= 0.05:
            alerts.append({"team": team, "edge": edge})
    return alerts


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": predictor.is_loaded()}


@app.get("/api/games")
def get_games(date: Date = Query(default=None)):
    """Return MLB games with live odds and ML predictions for a given date (defaults to today)."""
    try:
        games = odds_client.get_todays_mlb_games(date=date)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Odds API error: {e}")

    response = []
    for game in games:
        prediction = predictor.predict(game["home_team"], game["away_team"])
        alerts = _value_alerts(game["home_team"], game["away_team"], game.get("best_odds", {}), prediction)
        db.upsert_prediction(game, prediction, alerts)
        response.append({**game, "prediction": prediction})

    return {"games": response}


@app.get("/api/performance")
def get_performance():
    """Resolve any completed games then return prediction accuracy stats."""
    newly_resolved = results_module.resolve_pending(db)
    stats = db.get_performance_stats()
    return {**stats, "newly_resolved": newly_resolved}


@app.get("/api/odds/raw")
def get_raw_odds():
    """Raw odds response from The Odds API (for debugging)."""
    return odds_client.get_raw()
