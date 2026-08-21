"""
Case 1: Regional Airline x Fuel Price Shock — quantitative analysis
======================================================================
Answers the case's "Key Questions" with numbers instead of assumptions:
  1. How much of current losses are fuel-driven vs structurally embedded?
  2. Which routes are fundamentally broken vs salvageable?
  3. Scenario comparison of Option A (network optimization/pricing),
     Option B (UDAN expansion) and Option C (structural redesign).

Outputs 4 figures to reports/figures/ and a route-level verdict table
to data/processed/.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "airline_route_performance.csv"
FIG_DIR = ROOT / "reports" / "figures"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
PRE_SHOCK = "2024-10"


def load():
    df = pd.read_csv(DATA)
    df["month"] = pd.to_datetime(df["month"])
    df["period"] = np.where(df["month"] < PRE_SHOCK, "Pre-Shock (Apr-Sep24)", "Post-Shock (Oct24-Mar25)")
    return df


def fig1_fuel_price_vs_margin(df):
    monthly = df.groupby("month").agg(atf_price=("atf_price_per_litre", "mean"),
                                       ebitda_margin=("ebitda_margin", "mean")).reset_index()
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(monthly["month"], monthly["atf_price"], color="crimson", marker="o", label="ATF price (Rs/L)")
    ax2.plot(monthly["month"], monthly["ebitda_margin"] * 100, color="steelblue", marker="s", label="Avg EBITDA margin (%)")
    ax1.set_ylabel("ATF price (Rs/litre)", color="crimson")
    ax2.set_ylabel("Avg route EBITDA margin (%)", color="steelblue")
    ax1.set_title("Fuel Price Shock vs Network Profitability (FY24-25)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case1_fuel_price_vs_margin.png", dpi=140)
    plt.close(fig)


def fig2_loss_attribution(df):
    """Decompose post-shock losses: fuel-driven vs structural (non-fuel)."""
    pre = df[df["period"].str.startswith("Pre")].groupby("route_id").agg(
        pre_fuel_cost=("fuel_cost", "mean"), pre_ebitda=("ebitda", "mean")).reset_index()
    post = df[df["period"].str.startswith("Post")].groupby("route_id").agg(
        post_fuel_cost=("fuel_cost", "mean"), post_ebitda=("ebitda", "mean")).reset_index()
    merged = pre.merge(post, on="route_id")
    merged["fuel_driven_loss"] = -(merged["post_fuel_cost"] - merged["pre_fuel_cost"]).clip(lower=0)
    merged["total_ebitda_decline"] = merged["post_ebitda"] - merged["pre_ebitda"]
    merged["structural_component"] = merged["total_ebitda_decline"] - merged["fuel_driven_loss"]

    total_fuel = -merged["fuel_driven_loss"].sum()
    total_structural = -merged["structural_component"].clip(upper=0).sum()
    total_decline = -merged["total_ebitda_decline"].clip(upper=0).sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    shares = [total_fuel, max(total_structural, 0)]
    labels = [f"Fuel-driven\n(Rs {total_fuel/1e5:.1f}L)", f"Structural / other\n(Rs {max(total_structural,0)/1e5:.1f}L)"]
    ax.pie(shares, labels=labels, autopct="%1.0f%%", colors=["#d64550", "#4472ca"], startangle=90,
           wedgeprops=dict(edgecolor="white"))
    ax.set_title("Decline in Network EBITDA: Fuel-Driven vs Structural")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case1_loss_attribution.png", dpi=140)
    plt.close(fig)
    merged.to_csv(PROC_DIR / "case1_route_loss_attribution.csv", index=False)
    return merged


def fig3_route_viability(df):
    """Classify each route: Healthy / At-Risk / Structurally Unviable, post-shock."""
    post = df[df["period"].str.startswith("Post")].groupby(["route_id", "route", "is_udan_route"]).agg(
        avg_margin=("ebitda_margin", "mean"), avg_load_factor=("load_factor", "mean")).reset_index()

    def classify(row):
        if row.avg_margin >= 0.03:
            return "Healthy"
        elif row.avg_margin >= -0.05:
            return "At-Risk (thin/negative)"
        else:
            return "Structurally Unviable"

    post["verdict"] = post.apply(classify, axis=1)
    post = post.sort_values("avg_margin")
    post.to_csv(PROC_DIR / "case1_route_verdicts.csv", index=False)

    counts = post["verdict"].value_counts()
    pct_unviable = counts.get("Structurally Unviable", 0) / len(post)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = post["verdict"].map({"Healthy": "#2a9d5c", "At-Risk (thin/negative)": "#e9a325",
                                   "Structurally Unviable": "#d64550"})
    ax.barh(post["route"], post["avg_margin"] * 100, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Post-shock avg EBITDA margin (%)")
    ax.set_title(f"Route-Level Viability Post Fuel Shock  |  {pct_unviable:.0%} of routes structurally unviable")
    ax.tick_params(axis='y', labelsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case1_route_viability.png", dpi=140)
    plt.close(fig)
    return post, pct_unviable


def fig4_scenario_comparison():
    """Simple 18-24mo EBITDA-margin scenario model for Options A/B/C."""
    months = np.arange(0, 24)
    # illustrative recovery trajectories grounded in the case's stated
    # "time to impact" windows (A: 3-6mo, B: immediate, C: 12-24mo)
    option_a = -0.06 + 0.10 * (1 - np.exp(-months / 5))       # network optimization & pricing
    option_b = -0.02 + 0.015 * (months / 24)                  # UDAN expansion: quick stabilize, capped upside
    option_c = -0.09 + 0.19 * (1 / (1 + np.exp(-(months - 13) / 3)))  # structural redesign: slow, high payoff

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(months, option_a * 100, label="A: Network Optimization & Dynamic Pricing", linewidth=2.5)
    ax.plot(months, option_b * 100, label="B: Government-Backed Route Expansion (UDAN)", linewidth=2.5)
    ax.plot(months, option_c * 100, label="C: Structural Operating Model Redesign", linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Months from decision point")
    ax.set_ylabel("Projected network EBITDA margin (%)")
    ax.set_title("Illustrative Scenario Model: Strategic Options A vs B vs C")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case1_scenario_comparison.png", dpi=140)
    plt.close(fig)


def run():
    df = load()
    fig1_fuel_price_vs_margin(df)
    attribution = fig2_loss_attribution(df)
    verdicts, pct_unviable = fig3_route_viability(df)
    fig4_scenario_comparison()
    print(f"[Case 1] {pct_unviable:.0%} of routes structurally unviable post-shock")
    print(f"[Case 1] Figures written to {FIG_DIR}")
    return attribution, verdicts


if __name__ == "__main__":
    run()
