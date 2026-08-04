"""
GamePredictor / MultiPredictor — loads trained models and serves predictions.

Model detection in GamePredictor (force_model=None auto-detect):
  Priority: model_v2.pkl (pitcher-aware, 35 features)
         > model.pkl + dataset.pkl (v1 Bayesian)
         > model.pkl alone (v1 rolling)
         > Baseline: historical home-field win rate (~54%)

MultiPredictor loads ALL available models and returns predictions from each.
Use predict_all() to get a {model_key: prediction_dict} mapping.
"""

import os
import pickle
import sys
from datetime import datetime

import numpy as np

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "artifacts", "model.pkl")
MODEL_V2_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "model_v2.pkl")
DATASET_PATH  = os.path.join(os.path.dirname(__file__), "artifacts", "dataset.pkl")

_MLB_HOME_WIN_RATE = 0.54


class GamePredictor:
    """
    Wraps a single trained model.

    force_model: None  → auto-detect best available
                 "v2"  → load model_v2.pkl (pitcher-aware)
                 "v1"  → load model.pkl (rolling team stats)
    """

    def __init__(self, force_model=None):
        self._pipeline       = None
        self._model_version  = None
        self._final_features = None
        self._stats          = None
        self._stats_year     = None
        self._sp_stats       = None
        self._sp_stats_year  = None
        self._load_model(force_model)

    def _load_model(self, force_model):
        want_v2 = force_model == "v2" or (force_model is None and os.path.exists(MODEL_V2_PATH))
        want_v1 = force_model == "v1" or (force_model is None and not want_v2)

        if want_v2 and os.path.exists(MODEL_V2_PATH):
            with open(MODEL_V2_PATH, "rb") as f:
                artifact = pickle.load(f)
            self._pipeline      = artifact["pipeline"]
            self._model_version = "v2_pitcher"
            print(f"[predictor] loaded pitcher model (v2)", file=sys.stderr)
            return

        if want_v1 and os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, "rb") as f:
                self._pipeline = pickle.load(f)
            if os.path.exists(DATASET_PATH):
                with open(DATASET_PATH, "rb") as f:
                    data = pickle.load(f)
                self._final_features = data.get("final_features")
                self._model_version  = "v1_bayesian"
            else:
                self._model_version = "v1_rolling"
            print(f"[predictor] loaded {self._model_version} model", file=sys.stderr)

    def is_loaded(self):
        return self._pipeline is not None

    def predict(self, home_team, away_team,
                home_pitcher=None, away_pitcher=None):
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
                feats = self._get_features(home_team, away_team, home_pitcher, away_pitcher)
                if feats is not None:
                    proba = self._pipeline.predict_proba(feats.reshape(1, -1))[0]
                    home_prob = float(proba[1])
                    label_map = {
                        "v2_pitcher":  "ML Model v2 (pitcher)",
                        "v1_bayesian": "ML Model v1",
                        "v1_rolling":  "ML Model v1 (rolling)",
                    }
                    label = label_map.get(self._model_version, "ML Model")
                    return _format_prediction(home_team, away_team, home_prob, label)
                reason = "feature vector returned None"
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
        else:
            reason = "no model loaded"

        print(f"[predictor] falling back to baseline — {reason}", file=sys.stderr)
        return _format_prediction(
            home_team, away_team, _MLB_HOME_WIN_RATE, "Baseline (home field)"
        )

    def _get_features(self, home_team, away_team, home_pitcher=None, away_pitcher=None):
        year = datetime.now().year

        if self._model_version == "v2_pitcher":
            return self._get_features_v2(home_team, away_team, home_pitcher, away_pitcher, year)

        if self._model_version == "v1_bayesian":
            from model.serve_features import get_matchup_features
            return get_matchup_features(home_team, away_team, self._final_features, year)

        # v1_rolling
        from model.features import build_team_stats, game_features
        if self._stats is None or self._stats_year != year:
            self._stats      = build_team_stats(year)
            self._stats_year = year
        return game_features(home_team, away_team, self._stats)

    def _get_features_v2(self, home_team, away_team, home_pitcher, away_pitcher, year):
        from model.features import build_team_game_lookup

        if self._stats is None or self._stats_year != year:
            self._stats      = build_team_game_lookup(year)
            self._stats_year = year

        if not self._stats:
            return None

        _TEAM_FEATS = ["rpg", "rapg", "obp", "slg", "bb_pct", "k_pct", "wpct"]

        home_df = self._stats.get(home_team)
        away_df = self._stats.get(away_team)

        if home_df is None:
            for k, v in self._stats.items():
                if k[:3].upper() == home_team[:3].upper():
                    home_df = v
                    break
        if away_df is None:
            for k, v in self._stats.items():
                if k[:3].upper() == away_team[:3].upper():
                    away_df = v
                    break

        if home_df is None or away_df is None or home_df.empty or away_df.empty:
            return None

        try:
            home_team_feats = home_df.iloc[-1][_TEAM_FEATS].values.astype(float)
            away_team_feats = away_df.iloc[-1][_TEAM_FEATS].values.astype(float)
        except Exception:
            return None

        from model.pitcher_features import get_pitcher_season_stats, pitcher_feature_vector
        if self._sp_stats is None or self._sp_stats_year != year:
            self._sp_stats      = get_pitcher_season_stats(year)
            self._sp_stats_year = year

        home_sp_feats = pitcher_feature_vector(home_pitcher, self._sp_stats, year)
        away_sp_feats = pitcher_feature_vector(away_pitcher, self._sp_stats, year)

        feat_vec = np.concatenate([
            home_team_feats, away_team_feats,
            home_sp_feats, away_sp_feats,
            [1.0],
        ])

        if np.any(np.isnan(feat_vec)):
            return None

        return feat_vec


class MultiPredictor:
    """
    Loads all available model versions and returns predictions from each.
    Keys: "v2" (pitcher-aware) and/or "v1" (rolling team stats).
    """

    _KEY_V2 = "v2"
    _KEY_V1 = "v1"

    def __init__(self):
        self._predictors = {}
        if os.path.exists(MODEL_V2_PATH):
            self._predictors[self._KEY_V2] = GamePredictor(force_model="v2")
        if os.path.exists(MODEL_PATH):
            self._predictors[self._KEY_V1] = GamePredictor(force_model="v1")

    def is_loaded(self):
        return bool(self._predictors)

    def models(self):
        return list(self._predictors.keys())

    def predict_all(self, home_team, away_team,
                    home_pitcher=None, away_pitcher=None):
        """
        Returns {model_key: prediction_dict} for every loaded model.
        v2 receives pitcher names; v1 ignores them (no pitcher features).
        """
        results = {}
        for key, predictor in self._predictors.items():
            if key == self._KEY_V2:
                results[key] = predictor.predict(
                    home_team, away_team,
                    home_pitcher=home_pitcher, away_pitcher=away_pitcher,
                )
            else:
                results[key] = predictor.predict(home_team, away_team)
        return results

    def primary_prediction(self, all_predictions):
        """Best single prediction: v2 if available, else v1."""
        return (
            all_predictions.get(self._KEY_V2) or
            all_predictions.get(self._KEY_V1) or
            {}
        )


def _format_prediction(home_team, away_team, home_prob, model_used):
    away_prob = 1.0 - home_prob
    winner    = home_team if home_prob >= 0.5 else away_team
    win_prob  = max(home_prob, away_prob)

    if win_prob < 0.55:
        confidence = "low"
    elif win_prob < 0.65:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "home_win_prob":    round(home_prob, 3),
        "away_win_prob":    round(away_prob, 3),
        "predicted_winner": winner,
        "confidence":       confidence,
        "model_used":       model_used,
    }
