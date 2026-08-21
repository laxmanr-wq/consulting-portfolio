"""
Case 3: Telemedicine Missed Appointments — Intervention Simulation
========================================================================
This is the "Solution Design" + "Success Metrics" deliverable turned
into numbers. Given the case's constraints (You cannot: redesign the
whole product / assume new hardware. You can: improve existing
features, add small focused enhancements, modify flows/defaults),
this script simulates the effect of three concrete, low-lift product
levers using the EDA's observed no-show rates per segment:

  1. Upgrade default reminder to "SMS+Push" for users currently on
     "No Reminder" or "SMS"-only (a default/flow change, no new infra)
  2. Add a lightweight "connection test" nudge 10 min before slots for
     users with historically "Poor" connectivity (in-app interaction)
  3. Add a rebooking-friendly buffer for first-time users booked with
     >48h lead time (flow change: proactive reminder cadence)

Each lever's impact is estimated as: (baseline no-show rate for the
affected segment - observed no-show rate for users who already have
the "improved" attribute) x number of affected appointments. This is
the same before/after uplift logic a PM would use to size a feature
in a PRD, made explicit and reproducible rather than asserted.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "telemedicine_appointments.csv"
FIG_DIR = ROOT / "reports" / "figures"
PROC_DIR = ROOT / "data" / "processed"
sns.set_theme(style="whitegrid", palette="deep")

AVG_CONSULT_VALUE_RS = 450  # assumed avg platform take-rate per completed consult, for $-impact framing


def run():
    df = pd.read_csv(DATA)
    baseline_rate = df["no_show"].mean()

    levers = []

    # Lever 1: Reminder upgrade
    weak_reminder = df[df["reminder_channel"].isin(["No Reminder", "SMS", "Push"])]
    strong_reminder_rate = df[df["reminder_channel"] == "SMS+Push"]["no_show"].mean()
    weak_rate = weak_reminder["no_show"].mean()
    affected_1 = len(weak_reminder)
    uplift_1 = max(weak_rate - strong_reminder_rate, 0)
    appts_saved_1 = affected_1 * uplift_1
    levers.append(dict(
        lever="Default to SMS+Push reminders (currently No Reminder / SMS-only / Push-only)",
        affected_appointments=affected_1, baseline_rate=weak_rate,
        target_rate=strong_reminder_rate, appointments_saved=appts_saved_1,
        est_monthly_revenue_saved=appts_saved_1 * AVG_CONSULT_VALUE_RS,
    ))

    # Lever 2: Connectivity nudge for "Poor" connectivity users
    poor_conn = df[df["connectivity_quality"] == "Poor"]
    good_conn_rate = df[df["connectivity_quality"] == "Good"]["no_show"].mean()
    poor_rate = poor_conn["no_show"].mean()
    affected_2 = len(poor_conn)
    # Conservatively assume the nudge closes only half the gap to "Good" (can't fix underlying connectivity)
    uplift_2 = max((poor_rate - good_conn_rate) * 0.5, 0)
    appts_saved_2 = affected_2 * uplift_2
    levers.append(dict(
        lever="Pre-appointment connection-test nudge for 'Poor' connectivity users",
        affected_appointments=affected_2, baseline_rate=poor_rate,
        target_rate=poor_rate - uplift_2, appointments_saved=appts_saved_2,
        est_monthly_revenue_saved=appts_saved_2 * AVG_CONSULT_VALUE_RS,
    ))

    # Lever 3: Bookings made far in advance (>4 days out) are most likely to
    # be forgotten -- add a T-24h re-confirmation nudge for these bookings
    long_lead = df[df["lead_time_hours"] > 96]
    short_lead_rate = df[df["lead_time_hours"] <= 24]["no_show"].mean()
    long_lead_rate = long_lead["no_show"].mean()
    affected_3 = len(long_lead)
    uplift_3 = max(long_lead_rate - short_lead_rate, 0)
    appts_saved_3 = affected_3 * uplift_3
    levers.append(dict(
        lever="T-24h re-confirmation nudge for appointments booked >4 days in advance",
        affected_appointments=affected_3, baseline_rate=long_lead_rate,
        target_rate=short_lead_rate, appointments_saved=appts_saved_3,
        est_monthly_revenue_saved=appts_saved_3 * AVG_CONSULT_VALUE_RS,
    ))

    result = pd.DataFrame(levers)
    result["appointments_saved"] = result["appointments_saved"].round(0)
    result["est_monthly_revenue_saved"] = result["est_monthly_revenue_saved"].round(0)
    total_saved = result["appointments_saved"].sum()
    projected_new_rate = baseline_rate - total_saved / len(df)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(PROC_DIR / "case3_intervention_impact.csv", index=False)

    print(f"[Case 3 Simulation] Baseline no-show rate: {baseline_rate:.1%}")
    print(f"[Case 3 Simulation] Projected no-show rate after all 3 levers: {projected_new_rate:.1%}")
    print(result[["lever", "affected_appointments", "appointments_saved", "est_monthly_revenue_saved"]])

    # Chart: waterfall-style bar of appointments saved per lever
    fig, ax = plt.subplots(figsize=(9, 5.5))
    labels = [f"Lever {i+1}" for i in range(len(result))]
    ax.bar(labels, result["appointments_saved"], color=["#4472ca", "#e9a325", "#2a9d5c"])
    for i, v in enumerate(result["appointments_saved"]):
        ax.text(i, v + 2, f"{int(v)}", ha="center", fontweight="bold")
    ax.set_ylabel("Appointments saved (dataset window)")
    ax.set_title(f"Estimated Impact of 3 Product Levers\nProjected no-show rate: {baseline_rate:.1%} -> {projected_new_rate:.1%}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "case3_intervention_impact.png", dpi=140)
    plt.close(fig)

    return result, baseline_rate, projected_new_rate


if __name__ == "__main__":
    run()
