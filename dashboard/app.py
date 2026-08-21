"""
Consulting & Analytics Portfolio — Interactive Dashboard
============================================================
Run locally with:  streamlit run dashboard/app.py

A single Streamlit app that lets a reviewer (e.g. an HR/technical
interviewer) explore all three cases interactively: filter the
underlying data, see the same figures the notebooks produce, and read
the business recommendation for each case side by side with the
numbers behind it.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="Consulting & Analytics Portfolio", layout="wide", page_icon="📊")

st.title("📊 Consulting & Analytics Capstone — Portfolio Dashboard")
st.caption(
    "Three case studies (Winter Consulting Capstone, Consulting & Analytics Club, IIT Guwahati) "
    "solved end-to-end with real data pipelines, ML models, and quantified business recommendations."
)

case = st.sidebar.radio(
    "Choose a case",
    ["Overview", "Case 1 — Airline Fuel Shock", "Case 2 — OTT Viewer Retention", "Case 3 — Telemedicine No-Shows"],
)

# ----------------------------------------------------------------------
# OVERVIEW
# ----------------------------------------------------------------------
if case == "Overview":
    st.subheader("What this project covers")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### ✈️ Case 1: Airline")
        st.markdown(
            "Fuel-price shock hits a regional airline's route network. "
            "Quantifies fuel-driven vs structural losses, ranks route "
            "viability, and models 3 strategic options (A/B/C)."
        )
    with col2:
        st.markdown("### 🎬 Case 2: OTT Platform")
        st.markdown(
            "Viewers drop off mid-Season-1. A classifier + SHAP explain "
            "*why* (pacing, cognitive load), and K-Means segments "
            "episodes into actionable content clusters."
        )
    with col3:
        st.markdown("### 🩺 Case 3: Telemedicine")
        st.markdown(
            "Patients miss video consultations. A no-show risk model "
            "identifies who's at risk, and a simulation quantifies the "
            "impact of 3 concrete, low-lift product changes."
        )
    st.divider()
    st.markdown(
        "**Tech stack:** Python · pandas · scikit-learn · SHAP · matplotlib/seaborn · "
        "Plotly · Streamlit\n\n"
        "**Note on data:** the case brief supplied a real dataset only for Case 2, and that "
        "link isn't reachable outside the original competition context. All three datasets here "
        "are synthetically generated with realistic, documented unit-economics / behavioral "
        "assumptions (see `src/data_generation/`) so the full pipeline — EDA → modeling → "
        "business recommendation — can be run and verified end-to-end."
    )

# ----------------------------------------------------------------------
# CASE 1
# ----------------------------------------------------------------------
elif case.startswith("Case 1"):
    st.header("✈️ Case 1 — Regional Airline × Fuel Price Shock")
    df = pd.read_csv(RAW / "airline_route_performance.csv")
    df["month"] = pd.to_datetime(df["month"])

    verdicts = pd.read_csv(PROC / "case1_route_verdicts.csv")
    pct_unviable = (verdicts["verdict"] == "Structurally Unviable").mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Routes analyzed", len(verdicts))
    c2.metric("Structurally unviable post-shock", f"{pct_unviable:.0%}")
    c3.metric("Avg ATF price increase (Oct24→Mar25)", "+22%")

    st.subheader("Monthly network trend")
    monthly = df.groupby("month").agg(
        atf_price=("atf_price_per_litre", "mean"), ebitda_margin=("ebitda_margin", "mean")
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["atf_price"], name="ATF price (Rs/L)", yaxis="y1"))
    fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["ebitda_margin"] * 100, name="Avg EBITDA margin (%)", yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="ATF price (Rs/L)"),
        yaxis2=dict(title="EBITDA margin (%)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Route viability filter")
    udan_filter = st.selectbox("Route type", ["All", "UDAN-supported only", "Non-UDAN only"])
    v = verdicts.copy()
    if udan_filter == "UDAN-supported only":
        v = v[v["is_udan_route"] == True]
    elif udan_filter == "Non-UDAN only":
        v = v[v["is_udan_route"] == False]
    fig2 = px.bar(v.sort_values("avg_margin"), x="avg_margin", y="route", color="verdict",
                   orientation="h", height=600,
                   color_discrete_map={"Healthy": "#2a9d5c", "At-Risk (thin/negative)": "#e9a325",
                                        "Structurally Unviable": "#d64550"})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recommendation")
    st.info(
        "**Primary path: Option A (Network Optimization & Dynamic Pricing)** for the ~52% of "
        "routes still healthy or recoverable, **supported by Option B (selective UDAN "
        "expansion)** for the structurally unviable minority where subsidy makes the route "
        "viable at all. Option C (structural redesign) is flagged as a 12-24 month follow-on, "
        "not a primary response to an 18-24 month decision window."
    )

# ----------------------------------------------------------------------
# CASE 2
# ----------------------------------------------------------------------
elif case.startswith("Case 2"):
    st.header("🎬 Case 2 — OTT Viewer Retention")
    df = pd.read_csv(PROC / "case2_episodes_with_segments.csv")
    fi = pd.read_csv(PROC / "case2_feature_importance.csv", index_col=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Episodes analyzed", len(df))
    c2.metric("Top drop-off driver", fi.index[0].replace("_", " ").title())
    c3.metric("Worst episode (avg)", f"Ep {df.groupby('episode_number')['dropoff_rate'].mean().idxmax()}")

    st.subheader("Drop-off across Season 1")
    ep_stats = df.groupby("episode_number")["dropoff_rate"].mean().reset_index()
    fig = px.line(ep_stats, x="episode_number", y="dropoff_rate", markers=True)
    fig.add_vrect(x0=3.5, x1=5.5, fillcolor="orange", opacity=0.15, annotation_text="Mid-season slump")
    st.plotly_chart(fig, use_container_width=True)

    genre_filter = st.multiselect("Filter by genre", sorted(df["genre"].unique()), default=list(sorted(df["genre"].unique())))
    filtered = df[df["genre"].isin(genre_filter)]

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Drop-off driver importance (SHAP)")
        st.bar_chart(fi["mean_abs_shap"])
    with col_b:
        st.subheader("Segments by drop-off rate")
        seg_profile = filtered.groupby("segment")["dropoff_rate"].mean().sort_values()
        st.bar_chart(seg_profile)

    st.subheader("Recommendation")
    st.info(
        "**Reduce cognitive_load and increase pacing in episodes 4-5 specifically** — these two "
        "content-design levers dominate SHAP importance and coincide with the observed "
        "mid-season slump. Segment analysis further shows the highest-drop-off cluster is slow-"
        "paced, high-cognitive-load episodes with below-average watch-through — prioritize "
        "editing/pacing review for any future episode matching that profile before release."
    )

# ----------------------------------------------------------------------
# CASE 3
# ----------------------------------------------------------------------
else:
    st.header("🩺 Case 3 — Telemedicine Missed Appointments")
    df = pd.read_csv(RAW / "telemedicine_appointments.csv")
    impact = pd.read_csv(PROC / "case3_intervention_impact.csv")

    c1, c2, c3 = st.columns(3)
    c1.metric("Appointments analyzed", f"{len(df):,}")
    c2.metric("Baseline no-show rate", f"{df['no_show'].mean():.1%}")
    projected = df['no_show'].mean() - impact['appointments_saved'].sum() / len(df)
    c3.metric("Projected rate after 3 levers", f"{projected:.1%}", delta=f"-{df['no_show'].mean()-projected:.1%}")

    st.subheader("No-show rate by segment")
    dim = st.selectbox("Break down by", ["reminder_channel", "connectivity_quality", "time_slot", "specialty", "first_time_user"])
    seg = df.groupby(dim)["no_show"].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(seg, x=dim, y="no_show", color="no_show", color_continuous_scale="Reds")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Simulated impact of 3 product levers")
    st.dataframe(impact[["lever", "affected_appointments", "appointments_saved", "est_monthly_revenue_saved"]],
                 use_container_width=True, hide_index=True)
    fig2 = px.bar(impact, x="lever", y="appointments_saved", color="lever")
    fig2.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recommendation")
    st.info(
        "**Primary lever: make SMS+Push the default reminder configuration** (currently many "
        "users get no reminder or a single channel) — this alone accounts for the largest "
        "projected reduction and requires no new hardware or redesign, just a default/flow "
        "change, in line with the case's stated constraints. Supporting levers: a pre-call "
        "connectivity check for historically poor-connectivity users, and a T-24h re-"
        "confirmation nudge for appointments booked far in advance."
    )

st.sidebar.divider()
st.sidebar.caption("Built with pandas, scikit-learn, SHAP, Plotly & Streamlit.")
