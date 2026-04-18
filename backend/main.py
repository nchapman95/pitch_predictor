import os
from fastapi import FastAPI, HTTPException, Query
from datetime import date as Date
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from odds_client import OddsClient
from model.predictor import GamePredictor

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

    results = []
    for game in games:
        prediction = predictor.predict(game["home_team"], game["away_team"])
        results.append({
            **game,
            "prediction": prediction,
        })

    return {"games": results}


@app.get("/api/odds/raw")
def get_raw_odds():
    """Raw odds response from The Odds API (for debugging)."""
    return odds_client.get_raw()
