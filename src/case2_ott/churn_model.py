"""
Case 2: OTT Viewer Retention — Drop-off Risk Model
======================================================
Answers "Which factors have the greatest impact on drop-off risk?"
Trains a gradient-boosted classifier to predict whether an episode is
"high drop-off risk" (above platform median) and uses SHAP to explain
WHY -- turning the case's Key Question #1 into a ranked, defensible
answer instead of a guess.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay
import shap

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "ott_episode_engagement.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "pacing_score", "cognitive_load_score", "runtime_min",
    "cliffhanger_ending", "new_character_introduced", "recap_provided",
    "avg_watch_pct", "episode_number",
]


def load_and_prep():
    df = pd.read_csv(DATA)
    for c in ["cliffhanger_ending", "new_character_introduced", "recap_provided"]:
        df[c] = df[c].astype(int)
    X = df[FEATURES]
    y = df["dropoff_flag"]
    return df, X, y


def train():
    df, X, y = load_and_prep()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=False)
    print(f"[Case 2 Model] Test ROC-AUC: {auc:.3f}")
    print(report)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax, name="GradientBoosting")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title(f"High-Drop-off-Risk Episode Classifier (AUC={auc:.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case2_model_roc.png", dpi=140)
    plt.close(fig)

    # SHAP explainability -- this is the "why" behind the model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    plt.title("SHAP Feature Importance — What Drives Episode Drop-off Risk")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "case2_shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    mean_abs_shap.to_csv(ROOT / "data" / "processed" / "case2_feature_importance.csv", header=["mean_abs_shap"])
    print("\nTop drivers of drop-off risk (mean |SHAP|):")
    print(mean_abs_shap)

    return model, auc, mean_abs_shap


if __name__ == "__main__":
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    train()
