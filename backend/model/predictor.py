"""
GamePredictor: loads the trained model and serves predictions.

Model detection:
  - If artifacts/dataset.pkl exists → v1 model (392 Bayesian/rolling features).
    Features are computed live from artifacts/game_log_data/ parquets via
    serve_features.py.
  - Otherwise → v2 model (15 rolling features from features.py).
  - Falls back to historical home-field baseline (~54%) if no model is loaded.
"""

import os
import pickle
from datetime import datetime

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model.pkl")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "dataset.pkl")

_MLB_HOME_WIN_RATE = 0.54


class GamePredictor:
    def __init__(self):
        self._pipeline = None
        self._final_features = None   # set for v1 model
        self._stats = None            # used by v2 model
        self._stats_year = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self._pipeline = pickle.load(f)

        if os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, "rb") as f:
                data = pickle.load(f)
            self._final_features = data.get("final_features")

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def predict(self, home_team: str, away_team: str) -> dict:
        """
        Returns:
          {
            "home_win_prob": float,
            "away_win_prob": float,
            "predicted_winner": str,
            "confidence": str,   # "low" | "medium" | "high"
            "model_used": str,
          }
        """
        if self._pipeline is not None:
            try:
                feats = self._get_features(home_team, away_team)
                if feats is not None:
                    proba = self._pipeline.predict_proba(feats.reshape(1, -1))[0]
                    home_prob = float(proba[1])
                    label = "ML Model v1" if self._final_features else "ML Model v2"
                    return _format_prediction(home_team, away_team, home_prob, label)
                reason = "feature vector returned None"
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = "no model loaded"

        import sys
        print(f"[predictor] falling back to baseline — {reason}", file=sys.stderr)

        return _format_prediction(
            home_team, away_team, _MLB_HOME_WIN_RATE, "Baseline (home field)"
        )

    def _get_features(self, home_team: str, away_team: str):
        year = datetime.now().year

        if self._final_features is not None:
            # v1 model: Bayesian/rolling features from game log parquets
            from model.serve_features import get_matchup_features
            return get_matchup_features(home_team, away_team, self._final_features, year)

        # v2 model: rolling stats from the BRef scraper
        from model.features import build_team_stats, game_features
        if self._stats is None or self._stats_year != year:
            self._stats = build_team_stats(year)
            self._stats_year = year
        return game_features(home_team, away_team, self._stats)


def _format_prediction(
    home_team: str, away_team: str, home_prob: float, model_used: str
) -> dict:
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
