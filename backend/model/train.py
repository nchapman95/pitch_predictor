"""
Train an MLB game winner prediction model.

Usage:
    python -m model.train [--years 2019 2020 2021 2022 2023 2024]

Pipeline:
1. For each season, fetch per-game batting logs for every team via BRef
2. Compute rolling pre-game stats (last 15 games) for each team
3. For each home game, match both teams' pre-game stats by date
4. Train a GradientBoosting classifier (home win = 1)
5. Save model to model/artifacts/

Typical accuracy: ~57-60%
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd

import pybaseball
pybaseball.cache.enable()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from features import build_team_game_lookup, MIN_GAMES, feature_columns

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")

_TEAM_FEATS = ["rpg", "rapg", "obp", "slg", "bb_pct", "k_pct", "wpct"]


def build_training_data(years: list) -> tuple:
    X_all, y_all = [], []

    for year in years:
        print(f"Processing {year}...")
        lookup = build_team_game_lookup(year)
        if not lookup:
            print(f"  No data for {year}, skipping")
            continue

        # Build date -> row index for each team for fast lookup
        date_idx = {
            abb: {pd.Timestamp(row["Date"]).date(): i
                  for i, row in df.iterrows()}
            for abb, df in lookup.items()
        }

        games_added = 0
        for home_abb, home_df in lookup.items():
            home_games = home_df[home_df["Home"] == True]

            for _, row in home_games.iterrows():
                away_abb = str(row.get("Opp", "")).strip()
                if away_abb not in lookup:
                    continue

                game_date = pd.Timestamp(row["Date"]).date()
                h_idx = date_idx[home_abb].get(game_date)
                a_idx = date_idx[away_abb].get(game_date)
                if h_idx is None or a_idx is None:
                    continue

                # Require minimum game history for both teams
                if row["n_prior"] < MIN_GAMES:
                    continue
                a_row = lookup[away_abb].iloc[a_idx]
                if a_row["n_prior"] < MIN_GAMES:
                    continue

                home_feats = home_df.iloc[h_idx][_TEAM_FEATS].values.astype(float)
                away_feats = lookup[away_abb].iloc[a_idx][_TEAM_FEATS].values.astype(float)
                feat_vec = np.concatenate([home_feats, away_feats, [1.0]])

                if np.any(np.isnan(feat_vec)):
                    continue

                X_all.append(feat_vec)
                y_all.append(int(row["win"]))
                games_added += 1

        print(f"  {year}: added {games_added} games (total so far: {len(X_all)})")

    return np.array(X_all), np.array(y_all)


def train(years: list):
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
    print(f"CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

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
