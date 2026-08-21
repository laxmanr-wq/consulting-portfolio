"""
Synthetic data generator - Case 2: OTT Viewer Retention
==========================================================
The original case supplied a real dataset via a link that is not
retrievable in this environment, so a structurally equivalent
synthetic dataset is generated here. Column design mirrors exactly
what the case brief says the real dataset contains: "episode-level
viewing, behavioral, and content attributes... how episodes differ in
design, how viewers interact with them during playback, and the
resulting engagement outcomes."

The ground-truth relationship between content attributes and drop-off
is embedded deliberately (pacing_score, cognitive_load_score, and a
weak mid-season slump all push drop-off up) so downstream modeling
(feature importance / SHAP) has real signal to recover -- this is
what makes the "insights" in the notebook genuine rather than random.

Run:  python generate_ott_data.py
Output: data/raw/ott_episode_engagement.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(7)
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "ott_episode_engagement.csv"

GENRES = ["Thriller", "Drama", "Comedy", "Crime", "Romance", "SciFi"]
N_SERIES = 40
EPISODES_PER_SERIES = 8  # Season 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate():
    rows = []
    for series_id in range(1, N_SERIES + 1):
        genre = RNG.choice(GENRES)
        series_quality = RNG.normal(0, 1)          # unobserved "how good is the show" factor
        launch_viewers = int(RNG.uniform(80_000, 900_000))
        prev_retention = 1.0

        for ep in range(1, EPISODES_PER_SERIES + 1):
            pacing_score = np.clip(RNG.normal(6.5, 1.6), 1, 10)          # 1=very slow,10=very fast
            cognitive_load_score = np.clip(RNG.normal(5.5, 1.8), 1, 10)   # complexity of plot/characters
            runtime_min = RNG.normal(42, 8)
            cliffhanger_ending = RNG.random() < 0.45
            new_character_introduced = RNG.random() < (0.55 if ep <= 3 else 0.2)
            recap_provided = RNG.random() < 0.4
            avg_watch_pct = np.clip(RNG.normal(0.8, 0.12) - (ep == 4 or ep == 5) * 0.06, 0.15, 1.0)

            # ---- true generative model for drop-off probability ----
            slump = 1 if ep in (4, 5) else 0  # mid-season slump effect
            logit = (
                -1.4
                + 0.28 * (cognitive_load_score - 5.5)      # higher cognitive load -> more drop-off
                - 0.22 * (pacing_score - 6.5)                # faster pacing -> less drop-off
                + 0.55 * slump
                - 0.35 * cliffhanger_ending
                + 0.30 * new_character_introduced
                - 0.25 * recap_provided
                - 0.4 * series_quality
                + 0.10 * (runtime_min - 42) / 10
                - 0.6 * (avg_watch_pct - 0.8)
                + RNG.normal(0, 0.35)
            )
            dropoff_prob = sigmoid(logit)
            viewers_start = int(launch_viewers * prev_retention)
            viewers_completed = int(viewers_start * (1 - dropoff_prob))
            prev_retention = viewers_completed / launch_viewers if launch_viewers else 0

            rows.append(dict(
                series_id=series_id, genre=genre, episode_number=ep,
                pacing_score=round(pacing_score, 2),
                cognitive_load_score=round(cognitive_load_score, 2),
                runtime_min=round(runtime_min, 1),
                cliffhanger_ending=cliffhanger_ending,
                new_character_introduced=new_character_introduced,
                recap_provided=recap_provided,
                avg_watch_pct=round(avg_watch_pct, 3),
                viewers_start_of_episode=viewers_start,
                viewers_completed_episode=viewers_completed,
                dropoff_rate=round(dropoff_prob, 4),
            ))

    df = pd.DataFrame(rows)
    # Binary label a model would actually be trained on: is this episode's
    # drop-off rate above the platform-wide median (i.e. "high-risk" episode)?
    df["dropoff_flag"] = (df["dropoff_rate"] > df["dropoff_rate"].median()).astype(int)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows -> {OUT_PATH}")
    return df


if __name__ == "__main__":
    generate()
