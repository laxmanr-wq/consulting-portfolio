"""
Case 3: Telemedicine Missed Appointments — Exploratory Data Analysis
========================================================================
Grounds the case's "Problem Identification" requirement in actual
appointment-level patterns rather than surface-level symptoms.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "telemedicine_appointments.csv"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")


def load():
    return pd.read_csv(DATA)


def fig1_noshow_by_firsttime(df):
    stats = df.groupby("first_time_user")["no_show"].mean().reset_index()
    stats["first_time_user"] = stats["first_time_user"].map({True: "First-time user", False: "Returning user"})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=stats, x="first_time_user", y="no_show", hue="first_time_user",
                palette=["#d64550", "#2a9d5c"], legend=False, ax=ax)
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("")
    ax.set_title("No-Show Rate: First-Time vs Returning Users")
    for i, v in enumerate(stats["no_show"]):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case3_noshow_firsttime.png", dpi=140)
    plt.close(fig)


def fig2_noshow_by_reminder(df):
    order = ["No Reminder", "SMS", "Push", "SMS+Push", "SMS+Push+Call"]
    stats = df.groupby("reminder_channel")["no_show"].mean().reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=stats, x="reminder_channel", y="no_show", hue="reminder_channel",
                palette="rocket_r", legend=False, ax=ax)
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Reminder channel(s) used")
    ax.set_title("No-Show Rate by Reminder Strategy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case3_noshow_reminder.png", dpi=140)
    plt.close(fig)


def fig3_noshow_by_leadtime(df):
    df = df.copy()
    df["lead_time_bucket"] = pd.cut(
        df["lead_time_hours"], bins=[0, 2, 6, 24, 48, 96, 500],
        labels=["<2h", "2-6h", "6-24h", "1-2d", "2-4d", ">4d"])
    stats = df.groupby("lead_time_bucket", observed=True)["no_show"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=stats, x="lead_time_bucket", y="no_show", hue="lead_time_bucket",
                palette="mako", legend=False, ax=ax)
    ax.set_ylabel("No-show rate")
    ax.set_xlabel("Time between booking and appointment slot")
    ax.set_title("No-Show Rate by Booking Lead Time")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case3_noshow_leadtime.png", dpi=140)
    plt.close(fig)


def fig4_noshow_by_connectivity_slot(df):
    pivot = df.pivot_table(index="time_slot", columns="connectivity_quality", values="no_show", aggfunc="mean")
    pivot = pivot[["Poor", "Average", "Good"]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".1%", cmap="Reds", ax=ax, cbar_kws={"label": "No-show rate"})
    ax.set_title("No-Show Rate: Time Slot x Connectivity Quality")
    ax.set_xlabel("Connectivity quality")
    ax.set_ylabel("Time slot")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case3_noshow_heatmap.png", dpi=140)
    plt.close(fig)


def run():
    df = load()
    fig1_noshow_by_firsttime(df)
    fig2_noshow_by_reminder(df)
    fig3_noshow_by_leadtime(df)
    fig4_noshow_by_connectivity_slot(df)
    print(f"[Case 3 EDA] Overall no-show rate: {df['no_show'].mean():.1%}")
    print(f"[Case 3 EDA] Figures written to {FIG_DIR}")


if __name__ == "__main__":
    run()
