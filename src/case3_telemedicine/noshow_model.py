"""
Case 3: Telemedicine Missed Appointments — No-Show Risk Model
==================================================================
Builds a classifier a real product team would use to flag high-risk
appointments in real time (e.g. to trigger an extra reminder or a
priority reschedule offer) -- and uses SHAP to identify which levers
are within the PM's stated constraints (no redesign, no new hardware:
only flows/defaults/in-app interactions).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay
import shap

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "telemedicine_appointments.csv"
FIG_DIR = ROOT / "reports" / "figures"
PROC_DIR = ROOT / "data" / "processed"

NUMERIC = ["patient_age", "prior_completed_consults", "lead_time_hours", "doctor_reschedule_history"]
CATEGORICAL = ["specialty", "reminder_channel", "connectivity_quality", "time_slot"]
BOOLEAN = ["first_time_user", "fee_prepaid"]
FEATURES = NUMERIC + CATEGORICAL + BOOLEAN


def load_and_prep():
    df = pd.read_csv(DATA)
    for c in BOOLEAN:
        df[c] = df[c].astype(int)
    X = df[NUMERIC + CATEGORICAL + BOOLEAN]
    y = df["no_show"]
    return df, X, y


def train():
    df, X, y = load_and_prep()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="passthrough")

    pipe = Pipeline([
        ("pre", pre),
        ("clf", GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.04, random_state=42)),
    ])
    # No-show is imbalanced (~16% positive). GradientBoostingClassifier has
    # no class_weight param, so balance via sample_weight -- otherwise the
    # model just predicts "will show up" for everyone and looks falsely
    # accurate (this is a common real-world gotcha worth calling out in
    # an interview: accuracy alone is misleading on imbalanced targets).
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    sample_weight = np.where(y_train == 1, pos_weight, 1.0)
    pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    # Use a business-relevant threshold (flag top ~20% highest-risk
    # appointments) rather than the default 0.5, since the operational
    # question is "which appointments should get an extra nudge", not
    # "is this a 50/50 coin flip".
    threshold = np.quantile(y_proba, 0.80)
    y_pred = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    print(f"[Case 3 Model] Test ROC-AUC: {auc:.3f}  (flagging top 20% highest-risk appointments)")
    print(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="GradientBoosting")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_title(f"No-Show Risk Classifier (AUC={auc:.2f})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case3_model_roc.png", dpi=140)
    plt.close(fig)

    # SHAP on the transformed feature space
    feature_names = list(pipe.named_steps["pre"].get_feature_names_out())
    X_test_transformed = pipe.named_steps["pre"].transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_test_transformed)

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)
    top15 = mean_abs_shap.head(15)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top15.index[::-1], top15.values[::-1], color="#4472ca")
    ax.set_xlabel("Mean |SHAP value| (impact on no-show risk)")
    ax.set_title("Top Drivers of Appointment No-Show Risk")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case3_shap_importance.png", dpi=140)
    plt.close(fig)

    mean_abs_shap.to_csv(PROC_DIR / "case3_feature_importance.csv", header=["mean_abs_shap"])
    print("\nTop drivers of no-show risk (mean |SHAP|):")
    print(top15)

    return pipe, auc, mean_abs_shap


if __name__ == "__main__":
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train()
