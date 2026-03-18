"""
Train an MLB game winner prediction model.

Usage:
    python -m model.train [--years 2018 2019 2020 2021 2022 2023 2024]

Pipeline:
1. Pull historical game logs via pybaseball for each season
2. Pull team batting/pitching stats per season
3. Build feature vectors for each game
4. Train an XGBoost classifier (home win = 1)
5. Save model + scaler to model/artifacts/

Typical accuracy: ~57-60% (home field + team quality signals)
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

import pybaseball
pybaseball.cache.enable()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from features import build_team_stats, game_features, TEAM_NAME_TO_ABB

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")

# Reverse map: FG team name -> abbreviation
# pybaseball game logs use full city names; we map via standings/schedule
FG_TEAM_TO_ABB = {v: v for v in TEAM_NAME_TO_ABB.values()}  # identity for abbrevs


def _get_game_logs(year: int) -> pd.DataFrame:
    """Pull all game results for a season using pybaseball schedule_and_record."""
    frames = []
    for team_abb in list(set(TEAM_NAME_TO_ABB.values())):
        try:
            df = pybaseball.schedule_and_record(year, team_abb)
            df["team_abb"] = team_abb
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_training_data(years: list[int]) -> tuple[np.ndarray, np.ndarray]:
    X_all, y_all = [], []

    for year in years:
        print(f"Processing {year}...")
        stats = build_team_stats(year)
        if stats.empty:
            print(f"  No stats for {year}, skipping")
            continue

        logs = _get_game_logs(year)
        if logs.empty:
            continue

        # Filter to home games only (avoid duplicates)
        home_games = logs[logs["Home_Away"] == "Home"].copy()

        for _, row in home_games.iterrows():
            result = str(row.get("W/L", "")).strip().upper()
            if result not in ("W", "L"):
                continue

            home_team_abb = row["team_abb"]
            opp_abb = str(row.get("Opp", "")).strip()

            # We need full team names to look up in stats
            if home_team_abb not in stats.index or opp_abb not in stats.index:
                continue

            home_feats = stats.loc[home_team_abb].values.astype(float)
            away_feats = stats.loc[opp_abb].values.astype(float)
            feat_vec = np.concatenate([home_feats, away_feats, [1.0]])

            if np.any(np.isnan(feat_vec)):
                continue

            X_all.append(feat_vec)
            y_all.append(1 if result == "W" else 0)

    return np.array(X_all), np.array(y_all)


def train(years: list[int]):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Building training data...")
    X, y = build_training_data(years)
    print(f"Dataset: {len(X)} games, home win rate: {y.mean():.3f}")

    if len(X) == 0:
        print("No data collected. Make sure pybaseball can reach the internet.")
        return

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    pipeline.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2019, 2020, 2021, 2022, 2023, 2024],
    )
    args = parser.parse_args()
    train(args.years)
