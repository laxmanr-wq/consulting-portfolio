"""
Synthetic data generator - Case 3: Telemedicine Missed Appointments
========================================================================
No dataset was supplied for this case either (it's a pure product-
strategy brief). This generator creates an appointment-level event log
consistent with the constraints explicitly stated in the case:
  - a significant share of users are first-time telemedicine users
  - internet connectivity / device reliability varies
  - doctors operate on tight, fixed schedules
The ground-truth no-show model deliberately rewards realistic product
levers (lead time, reminder count/channel, first-time-user flag,
connectivity quality, time-of-day) so the notebook's model + SHAP
analysis recovers levers a PM could actually act on -- directly
mirroring the case's "Your Role" constraints (no redesign, no new
hardware, only flow/default/in-app changes).

Run:  python generate_telemedicine_data.py
Output: data/raw/telemedicine_appointments.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(21)
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "telemedicine_appointments.csv"

N = 12000
SPECIALTIES = ["General Physician", "Dermatology", "Pediatrics", "Psychiatry", "Gynecology", "ENT"]
REMINDER_CHANNELS = ["No Reminder", "SMS", "Push", "SMS+Push", "SMS+Push+Call"]  # "No Reminder" (not "None") avoids pandas NA-string parsing on CSV read
CONNECTIVITY = ["Poor", "Average", "Good"]
TIME_SLOTS = ["Early Morning (7-9)", "Morning (9-12)", "Afternoon (12-4)", "Evening (4-8)", "Night (8-10)"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate():
    ids = np.arange(1, N + 1)
    first_time_user = RNG.random(N) < 0.38
    lead_time_hours = np.round(np.clip(RNG.exponential(30, N), 0.5, 240), 1)  # time between booking & slot
    reminder_channel = RNG.choice(REMINDER_CHANNELS, N, p=[0.15, 0.30, 0.20, 0.25, 0.10])
    connectivity_quality = RNG.choice(CONNECTIVITY, N, p=[0.18, 0.42, 0.40])
    time_slot = RNG.choice(TIME_SLOTS, N, p=[0.08, 0.34, 0.32, 0.20, 0.06])
    specialty = RNG.choice(SPECIALTIES, N)
    prior_completed_consults = RNG.poisson(2.2, N)
    prior_completed_consults = np.where(first_time_user, 0, prior_completed_consults)
    doctor_reschedule_history = RNG.poisson(0.4, N)   # # times this doctor's slots were rescheduled before
    fee_prepaid = RNG.random(N) < 0.30                 # most consults are pay-after-session per case brief
    age = np.clip(RNG.normal(38, 14, N), 5, 85).round(0)

    reminder_strength = pd.Series(reminder_channel).map({
        "No Reminder": 0, "SMS": 1, "Push": 1, "SMS+Push": 2, "SMS+Push+Call": 3
    }).values
    connectivity_strength = pd.Series(connectivity_quality).map({"Poor": 0, "Average": 1, "Good": 2}).values
    slot_risk = pd.Series(time_slot).map({
        "Early Morning (7-9)": 0.35, "Morning (9-12)": -0.05, "Afternoon (12-4)": -0.15,
        "Evening (4-8)": 0.05, "Night (8-10)": 0.30,
    }).values

    # ---- true generative model for no-show probability ----
    logit = (
        -1.1
        + 0.55 * first_time_user
        + 0.012 * np.clip(lead_time_hours - 24, -24, 200) / 10   # very long lead time -> more forgetting
        - 0.38 * reminder_strength
        - 0.30 * connectivity_strength
        + slot_risk
        - 0.10 * prior_completed_consults
        + 0.25 * doctor_reschedule_history
        - 0.20 * fee_prepaid
        + RNG.normal(0, 0.4, N)
    )
    noshow_prob = sigmoid(logit)
    noshow = (RNG.random(N) < noshow_prob).astype(int)
    joined_late_minutes = np.where(
        noshow == 0,
        np.clip(RNG.exponential(2.5, N) - 1.0 * connectivity_strength, 0, None).round(1),
        np.nan,
    )

    df = pd.DataFrame(dict(
        appointment_id=ids,
        patient_age=age.astype(int),
        specialty=specialty,
        first_time_user=first_time_user,
        prior_completed_consults=prior_completed_consults,
        lead_time_hours=lead_time_hours,
        reminder_channel=reminder_channel,
        connectivity_quality=connectivity_quality,
        time_slot=time_slot,
        doctor_reschedule_history=doctor_reschedule_history,
        fee_prepaid=fee_prepaid,
        joined_late_minutes=joined_late_minutes,
        no_show=noshow,
    ))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows -> {OUT_PATH}  |  no-show rate: {df['no_show'].mean():.1%}")
    return df


if __name__ == "__main__":
    generate()
