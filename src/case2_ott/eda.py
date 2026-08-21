"""
Case 2: OTT Viewer Retention — Exploratory Data Analysis
============================================================
Answers "What patterns/relationships explain differences in viewer
continuation across episodes?" with visuals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "ott_episode_engagement.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")


def load():
    return pd.read_csv(DATA)


def fig1_dropoff_by_episode(df):
    ep_stats = df.groupby("episode_number")["dropoff_rate"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ep_stats["episode_number"], ep_stats["dropoff_rate"] * 100, marker="o",
            linewidth=2.5, color="#d64550")
    ax.fill_between([3.5, 5.5], 0, ep_stats["dropoff_rate"].max() * 105, alpha=0.15, color="orange",
                     label="Mid-season slump (Ep 4-5)")
    ax.set_xlabel("Episode number (Season 1)")
    ax.set_ylabel("Avg episode drop-off rate (%)")
    ax.set_title("Viewer Drop-off Rate Across Season 1 — the 'Mid-Season Slump'")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case2_dropoff_by_episode.png", dpi=140)
    plt.close(fig)


def fig2_correlation_heatmap(df):
    numeric_cols = ["pacing_score", "cognitive_load_score", "runtime_min", "avg_watch_pct", "dropoff_rate"]
    bool_cols = ["cliffhanger_ending", "new_character_introduced", "recap_provided"]
    corr_df = df[numeric_cols].copy()
    for c in bool_cols:
        corr_df[c] = df[c].astype(int)
    corr = corr_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Correlation: Content Design Attributes vs Drop-off Rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case2_correlation_heatmap.png", dpi=140)
    plt.close(fig)


def fig3_pacing_cogload_scatter(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df["pacing_score"], df["cognitive_load_score"], c=df["dropoff_rate"] * 100,
                     cmap="RdYlGn_r", s=45, alpha=0.75, edgecolor="white", linewidth=0.3)
    cbar = fig.colorbar(sc)
    cbar.set_label("Drop-off rate (%)")
    ax.set_xlabel("Pacing score (1=slow, 10=fast)")
    ax.set_ylabel("Cognitive load score (1=simple, 10=complex)")
    ax.set_title("Episode Drop-off Risk by Pacing vs Cognitive Load")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case2_pacing_cogload_scatter.png", dpi=140)
    plt.close(fig)


def fig4_genre_comparison(df):
    genre_stats = df.groupby("genre")["dropoff_rate"].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=genre_stats, x="dropoff_rate", y="genre", hue="genre", ax=ax, palette="rocket", legend=False)
    ax.set_xlabel("Avg drop-off rate")
    ax.set_title("Drop-off Rate by Genre")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case2_genre_comparison.png", dpi=140)
    plt.close(fig)


def run():
    df = load()
    fig1_dropoff_by_episode(df)
    fig2_correlation_heatmap(df)
    fig3_pacing_cogload_scatter(df)
    fig4_genre_comparison(df)
    print(f"[Case 2 EDA] Figures written to {FIG_DIR}")


if __name__ == "__main__":
    run()
