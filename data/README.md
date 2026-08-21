# Data Dictionary

All three raw datasets in this folder are **synthetically generated** by the
scripts in `src/data_generation/`. See the main [README](../README.md) for
why, and each script's docstring for exactly how.

## `airline_route_performance.csv` (Case 1)
48 routes × 12 months (FY24–FY25), one row per route-month.

| Column | Description |
|---|---|
| `route_id`, `route` | Route identifier / city-pair name |
| `seats`, `weekly_frequency`, `block_hours` | Aircraft/schedule config |
| `is_udan_route` | Whether route is UDAN-subsidy eligible |
| `atf_price_per_litre` | Aviation Turbine Fuel price that month |
| `load_factor` | Seats filled, 0–1 |
| `avg_fare`, `passengers` | Avg ticket price / monthly passengers |
| `ticket_revenue`, `ancillary_revenue`, `udan_subsidy` | Revenue components |
| `fuel_cost`, `lease_crew_cost`, `airport_handling_cost` | Cost components |
| `total_revenue`, `total_cost`, `ebitda`, `ebitda_margin` | Derived P&L |

## `ott_episode_engagement.csv` (Case 2)
40 series × 8 episodes (Season 1), one row per series-episode.

| Column | Description |
|---|---|
| `series_id`, `genre`, `episode_number` | Identifiers |
| `pacing_score` | 1 (slow) – 10 (fast) |
| `cognitive_load_score` | 1 (simple) – 10 (complex plot/characters) |
| `runtime_min` | Episode runtime |
| `cliffhanger_ending`, `new_character_introduced`, `recap_provided` | Content design flags |
| `avg_watch_pct` | Avg % of episode watched by viewers who started it |
| `viewers_start_of_episode`, `viewers_completed_episode` | Raw viewer counts |
| `dropoff_rate` | Probability a viewer doesn't return for the next episode |
| `dropoff_flag` | 1 if `dropoff_rate` above platform median (model target) |

## `telemedicine_appointments.csv` (Case 3)
12,000 appointment records.

| Column | Description |
|---|---|
| `appointment_id`, `patient_age`, `specialty` | Identifiers |
| `first_time_user` | Whether this is the patient's first telemedicine booking |
| `prior_completed_consults` | Consult history (0 for first-time users) |
| `lead_time_hours` | Hours between booking and the scheduled slot |
| `reminder_channel` | `No Reminder` / `SMS` / `Push` / `SMS+Push` / `SMS+Push+Call` |
| `connectivity_quality` | Patient's typical connection quality: Poor/Average/Good |
| `time_slot` | Scheduled time-of-day bucket |
| `doctor_reschedule_history` | # times this doctor's slots were rescheduled before |
| `fee_prepaid` | Whether payment was collected upfront (most are pay-after-session) |
| `joined_late_minutes` | Minutes late to join, if they showed up (NaN if no-show) |
| `no_show` | 1 if patient did not join (model target) |

> ⚠️ Note the `reminder_channel` value is `"No Reminder"`, not the string
> `"None"` — a real bug caught during development where pandas silently
> parses a literal `"None"` string as `NaN` on CSV read. See
> `tests/test_data_pipeline.py::test_reminder_channel_no_pandas_na_collision`.
