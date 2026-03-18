"""
GamePredictor: loads the trained model and serves predictions.

If the model hasn't been trained yet, returns a fallback based on
home-field advantage (~54% historical MLB home win rate).
"""

import os
import pickle
import numpy as np
from datetime import datetime

from model.features import build_team_stats, game_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model.pkl")

_MLB_HOME_WIN_RATE = 0.54  # historical baseline


class GamePredictor:
    def __init__(self):
        self._pipeline = None
        self._stats = None
        self._stats_year = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self._pipeline = pickle.load(f)

    def _get_stats(self) -> object:
        year = datetime.now().year
        if self._stats is None or self._stats_year != year:
            self._stats = build_team_stats(year)
            self._stats_year = year
        return self._stats

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def predict(self, home_team: str, away_team: str) -> dict:
        """
        Returns:
          {
            "home_win_prob": float,   # 0-1
            "away_win_prob": float,
            "predicted_winner": str,
            "confidence": str,        # "low" | "medium" | "high"
            "model_used": str,
          }
        """
        if self._pipeline is not None:
            try:
                stats = self._get_stats()
                feats = game_features(home_team, away_team, stats)
                if feats is not None:
                    proba = self._pipeline.predict_proba(feats.reshape(1, -1))[0]
                    home_prob = float(proba[1])
                    return _format_prediction(home_team, away_team, home_prob, "ML Model")
            except Exception:
                pass  # fall through to baseline

        # Baseline: home field advantage
        return _format_prediction(home_team, away_team, _MLB_HOME_WIN_RATE, "Baseline (home field)")


def _format_prediction(home_team: str, away_team: str, home_prob: float, model_used: str) -> dict:
    away_prob = 1.0 - home_prob
    winner = home_team if home_prob >= 0.5 else away_team
    win_prob = max(home_prob, away_prob)

    if win_prob < 0.55:
        confidence = "low"
    elif win_prob < 0.65:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "home_win_prob": round(home_prob, 3),
        "away_win_prob": round(away_prob, 3),
        "predicted_winner": winner,
        "confidence": confidence,
        "model_used": model_used,
    }
