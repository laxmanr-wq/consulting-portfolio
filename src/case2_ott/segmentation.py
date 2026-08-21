"""
Case 2: OTT Viewer Retention — Episode Segmentation
=======================================================
Answers "Can episodes be meaningfully segmented based on content
characteristics and viewer behavior, and how do these segments differ
in engagement outcomes?"

Uses K-Means (k chosen via silhouette score) on standardized content +
behavior features, then profiles each cluster's drop-off outcome so
each segment maps to a concrete recommendation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "ott_episode_engagement.csv"
FIG_DIR = ROOT / "reports" / "figures"
PROC_DIR = ROOT / "data" / "processed"
sns.set_theme(style="whitegrid", palette="deep")

CLUSTER_FEATURES = ["pacing_score", "cognitive_load_score", "runtime_min", "avg_watch_pct"]


def choose_k(X_scaled, k_range=range(2, 5)):
    # Capped at 5: silhouette keeps rising marginally beyond this, but a
    # 2-4 segment story is the one a business stakeholder can actually
    # act on -- more clusters than that stops being decision-useful.
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        scores[k] = silhouette_score(X_scaled, km.labels_)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def run():
    df = pd.read_csv(DATA)
    X = df[CLUSTER_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, scores = choose_k(X_scaled)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(list(scores.keys()), list(scores.values()), marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=f"chosen k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Choosing k for Episode Segmentation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case2_silhouette.png", dpi=140)
    plt.close(fig)

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X_scaled)
    df["segment"] = km.labels_

    # Profile clusters
    profile = df.groupby("segment").agg(
        n_episodes=("segment", "size"),
        avg_pacing=("pacing_score", "mean"),
        avg_cognitive_load=("cognitive_load_score", "mean"),
        avg_runtime=("runtime_min", "mean"),
        avg_watch_pct=("avg_watch_pct", "mean"),
        avg_dropoff_rate=("dropoff_rate", "mean"),
    ).round(2).sort_values("avg_dropoff_rate")
    profile.to_csv(PROC_DIR / "case2_segment_profile.csv")
    print(f"[Case 2 Segmentation] best_k={best_k}")
    print(profile)

    # PCA 2D visualization of segments
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df["pc1"], df["pc2"] = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    scatter = sns.scatterplot(data=df, x="pc1", y="pc2", hue="segment", palette="Set2", s=70,
                               edgecolor="white", linewidth=0.4, ax=ax)
    ax.set_title(f"Episode Segments (PCA projection, k={best_k})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} var)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case2_segments_pca.png", dpi=140)
    plt.close(fig)

    # Segment vs drop-off bar
    fig, ax = plt.subplots(figsize=(7, 5))
    order = profile.index.tolist()
    sns.barplot(data=df, x="segment", y="dropoff_rate", order=order, hue="segment",
                palette="rocket", legend=False, ax=ax)
    ax.set_title("Drop-off Rate by Episode Segment")
    ax.set_ylabel("Avg drop-off rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case2_segments_dropoff.png", dpi=140)
    plt.close(fig)

    df.to_csv(PROC_DIR / "case2_episodes_with_segments.csv", index=False)
    return df, profile


if __name__ == "__main__":
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    run()
