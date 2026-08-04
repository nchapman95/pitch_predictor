"""
Train the v2 MLB game winner model — adds starting pitcher features.

Usage:
    python -m model.train_v2 [--years 2021 2022 2023 2024 2025]
    python -m model.train_v2 --no-rolling   # skip rolling stats (faster)

Pipeline:
1. For each season, build team rolling stats (same as v1 via features.py)
2. Fetch all game dates + probable starters from MLB Stats API
3. Fetch pitcher season stats from FanGraphs via pybaseball
4. Build combined feature vector per game:
     [home_team_feats(7), away_team_feats(7), home_sp_feats(10), away_sp_feats(10), home_field(1)]
   = 35 features  (vs 15 in v1)
5. Train GradientBoosting classifier; save to artifacts/model_v2.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

import pybaseball
pybaseball.cache.enable()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from features import build_team_game_lookup, MIN_GAMES, ROLLING_WINDOW
from pitcher_features import (
    get_pitcher_season_stats,
    get_starters_for_date,
    pitcher_feature_vector,
    pitcher_feature_columns,
    N_SP_FEATS,
)

ARTIFACTS_DIR       = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH          = os.path.join(ARTIFACTS_DIR, "model_v2.pkl")
MLFLOW_TRACKING_URI = os.path.join(os.path.dirname(__file__), "mlruns")
MLFLOW_EXPERIMENT   = "mlb-pitcher-model"
MLFLOW_MODEL_NAME   = "mlb-game-predictor-v2"

_TEAM_FEATS = ["rpg", "rapg", "obp", "slg", "bb_pct", "k_pct", "wpct"]


def _build_starter_cache(year: int, dates: list) -> dict:
    """
    Fetch probable/actual starters for all game dates in a season.
    Returns {date_str: {matchup_key: {home_pitcher, away_pitcher, ...}}}
    """
    cache = {}
    total = len(dates)
    for i, date_str in enumerate(dates):
        if i % 10 == 0:
            print(f"  Fetching starters {i}/{total}...", end="\r")
        cache[date_str] = get_starters_for_date(date_str)
    print(f"  Fetched starters for {total} dates            ")
    return cache


def build_training_data(years: list, include_rolling: bool = True) -> tuple:
    X_all, y_all = [], []

    for year in years:
        print(f"\nProcessing {year}...")

        # --- Team rolling stats ---
        team_lookup = build_team_game_lookup(year)
        if not team_lookup:
            print(f"  No team data for {year}, skipping")
            continue

        date_idx = {
            abb: {pd.Timestamp(row["Date"]).date(): i for i, row in df.iterrows()}
            for abb, df in team_lookup.items()
        }

        # --- Starter cache (one MLB Stats API call per date) ---
        all_dates = sorted({
            str(pd.Timestamp(row["Date"]).date())
            for df in team_lookup.values()
            for _, row in df.iterrows()
        })
        starter_cache = _build_starter_cache(year, all_dates)

        # --- Point-in-time pitcher stat snapshots (one call per unique date) ---
        # Cached by date so we don't re-fetch for every game on the same day.
        # through_date is passed to prevent any forward-looking data leakage.
        pitcher_stats_cache: dict[str, pd.DataFrame] = {}
        print(f"  Will fetch pitcher stats snapshots for {len(all_dates)} dates (cached per date)")

        games_added = 0
        for home_abb, home_df in team_lookup.items():
            home_games = home_df[home_df["Home"] == True]

            for _, row in home_games.iterrows():
                away_abb = str(row.get("Opp", "")).strip()
                if away_abb not in team_lookup:
                    continue

                result = str(row.get("W/L", "")).strip().upper()
                if result not in ("W", "L"):
                    continue

                game_date = pd.Timestamp(row["Date"]).date()
                h_idx = date_idx[home_abb].get(game_date)
                a_idx = date_idx[away_abb].get(game_date)
                if h_idx is None or a_idx is None:
                    continue

                if row["n_prior"] < MIN_GAMES:
                    continue
                a_row = team_lookup[away_abb].iloc[a_idx]
                if a_row["n_prior"] < MIN_GAMES:
                    continue

                # Team features
                home_team_feats = home_df.iloc[h_idx][_TEAM_FEATS].values.astype(float)
                away_team_feats = team_lookup[away_abb].iloc[a_idx][_TEAM_FEATS].values.astype(float)

                # Pitcher features — point-in-time stats (no future leakage)
                date_str = str(game_date)
                day_starters = starter_cache.get(date_str, {})

                # Find the right matchup in the starter cache
                home_pitcher = away_pitcher = None
                for matchup_data in day_starters.values():
                    if (matchup_data.get("home_team", "")[:3].upper() == home_abb[:3].upper() or
                            matchup_data.get("away_team", "")[:3].upper() == away_abb[:3].upper()):
                        home_pitcher = matchup_data.get("home_pitcher")
                        away_pitcher = matchup_data.get("away_pitcher")
                        break

                # Fetch pitcher stats as of this game date (cached per date)
                if date_str not in pitcher_stats_cache:
                    pitcher_stats_cache[date_str] = get_pitcher_season_stats(
                        year, through_date=date_str
                    )
                pit_stats = pitcher_stats_cache[date_str]

                home_sp_feats = pitcher_feature_vector(
                    home_pitcher, pit_stats, year, include_rolling=include_rolling
                )
                away_sp_feats = pitcher_feature_vector(
                    away_pitcher, pit_stats, year, include_rolling=include_rolling
                )

                feat_vec = np.concatenate([
                    home_team_feats, away_team_feats,
                    home_sp_feats, away_sp_feats,
                    [1.0],
                ])

                if np.any(np.isnan(feat_vec)):
                    continue

                X_all.append(feat_vec)
                y_all.append(int(result == "W"))
                games_added += 1

        print(f"  {year}: {games_added} games added  (total so far: {len(X_all)})")

    return np.array(X_all), np.array(y_all)


def feature_columns_v2() -> list[str]:
    team_feats = _TEAM_FEATS
    return (
        [f"home_{c}" for c in team_feats] +
        [f"away_{c}" for c in team_feats] +
        pitcher_feature_columns() +
        ["home_field"]
    )


def train(years: list, include_rolling: bool = True,
          n_estimators: int = 300, max_depth: int = 4,
          learning_rate: float = 0.05, subsample: float = 0.8):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(tags={"model_id": MLFLOW_MODEL_NAME}):
        params = {
            "years": str(years),
            "n_years": len(years),
            "include_rolling": include_rolling,
            "rolling_window": ROLLING_WINDOW,
            "min_games": MIN_GAMES,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "n_pitcher_feats": N_SP_FEATS,
        }
        mlflow.log_params(params)

        print("Building training data with pitcher features...")
        X, y = build_training_data(years, include_rolling=include_rolling)

        if len(X) == 0:
            print("No training data collected.")
            return

        print(f"\nDataset: {len(X)} games  |  {X.shape[1]} features  |  home win rate: {y.mean():.3f}")
        print(f"Feature columns ({len(feature_columns_v2())}):")
        for col in feature_columns_v2():
            print(f"  {col}")

        mlflow.log_metrics({
            "n_games": len(X),
            "home_win_rate": float(y.mean()),
            "n_features": X.shape[1],
        })

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=42,
            )),
        ])

        print("\nRunning 5-fold cross-validation...")
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
        print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

        mlflow.log_metrics({
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            **{f"cv_fold_{i+1}": float(s) for i, s in enumerate(scores)},
        })

        pipeline.fit(X, y)
        train_acc = float(pipeline.score(X, y))
        print(f"Train accuracy: {train_acc:.3f}")
        mlflow.log_metric("train_accuracy", train_acc)

        mlflow.sklearn.log_model(pipeline, "model",
                                 registered_model_name=MLFLOW_MODEL_NAME)

        # Save model + metadata as pickle (for live serving)
        artifact = {
            "pipeline": pipeline,
            "feature_columns": feature_columns_v2(),
            "n_features": X.shape[1],
            "years_trained": years,
            "include_rolling": include_rolling,
            "mlflow_run_id": mlflow.active_run().info.run_id,
            "model_id": MLFLOW_MODEL_NAME,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(artifact, f)
        mlflow.log_artifact(MODEL_PATH, "pickle")

        print(f"\nModel saved to {MODEL_PATH}")
        print(f"MLflow experiment : {MLFLOW_EXPERIMENT}")
        print(f"MLflow model name : {MLFLOW_MODEL_NAME}")
        print(f"MLflow run ID     : {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--no-rolling", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    args = parser.parse_args()
    train(
        years=args.years,
        include_rolling=not args.no_rolling,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
    )
