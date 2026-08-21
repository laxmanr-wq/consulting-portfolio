"""
Synthetic data generator - Case 1: Regional Airline x Fuel Price Shock
========================================================================
No real dataset was supplied for this case (it's a strategy/consulting
case, not a data case). This script generates a realistic, internally
consistent route-economics dataset so the problem can be analyzed
quantitatively instead of just qualitatively.

The generation logic is grounded in real aviation unit-economics:
  revenue  = seats * load_factor * avg_fare  + ancillary_revenue
  cost     = fuel_cost + fixed_lease_crew_cost + airport_handling_cost
  fuel_cost = block_hours * fuel_burn_per_hour * ATF_price_per_litre
Margins, load factors and UDAN-subsidy eligibility are sampled from
distributions that mirror the FY24/FY25 figures described in the case
brief (load factors 70-75%, ATF as largest variable cost, etc.)

Run:  python generate_airline_data.py
Output: data/raw/airline_route_performance.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "airline_route_performance.csv"

N_ROUTES = 48
CITY_PAIRS_TIER2_3 = [
    "Kharagpur-Kolkata", "Jamshedpur-Bhubaneswar", "Nashik-Pune", "Kanpur-Lucknow",
    "Coimbatore-Chennai", "Indore-Bhopal", "Rajkot-Ahmedabad", "Guwahati-Dibrugarh",
    "Varanasi-Delhi", "Madurai-Bengaluru", "Raipur-Nagpur", "Jodhpur-Jaipur",
    "Amritsar-Chandigarh", "Vizag-Hyderabad", "Patna-Ranchi", "Surat-Mumbai",
]

months = pd.date_range("2024-04-01", "2025-03-01", freq="MS")  # FY24-FY25


def generate():
    rows = []
    for route_id in range(1, N_ROUTES + 1):
        pair = CITY_PAIRS_TIER2_3[route_id % len(CITY_PAIRS_TIER2_3)] + f"-R{route_id}"
        seats = int(RNG.choice([78, 90, 108, 138]))            # narrow-body regional config
        block_hours = round(RNG.uniform(0.8, 2.2), 2)          # short-haul
        fuel_burn_per_hr = RNG.uniform(1600, 1950)               # litres/hr, regional turboprop/jet
        is_udan = RNG.random() < 0.35                          # ~35% routes are UDAN-supported
        base_fare = RNG.uniform(2200, 5800)
        weekly_frequency = int(RNG.choice([3, 5, 7, 10, 14]))

        for month in months:
            # ATF price shock: rises through FY25 (Oct24 - Mar25) due to crude volatility
            month_idx = (month.year - 2024) * 12 + month.month
            fy25_shock_start = (2024 - 2024) * 12 + 10  # Oct 2024 onward = shock
            atf_price = 92 if month < pd.Timestamp("2024-10-01") else 92 * (1 + RNG.uniform(0.18, 0.34))
            atf_price += RNG.normal(0, 3)  # noise

            load_factor = np.clip(RNG.normal(0.725, 0.06) - (0.04 if is_udan is False and atf_price > 100 else 0), 0.45, 0.95)
            avg_fare = base_fare * (1 + RNG.normal(0, 0.05)) * (1.06 if atf_price > 110 else 1.0)  # partial fare pass-through

            flights_month = weekly_frequency * 4.33
            pax_per_flight = seats * load_factor
            pax_month = pax_per_flight * flights_month
            ticket_revenue = pax_month * avg_fare
            ancillary_revenue = pax_month * RNG.uniform(180, 420)
            udan_subsidy = ticket_revenue * RNG.uniform(0.12, 0.22) if is_udan else 0

            fuel_cost = flights_month * block_hours * fuel_burn_per_hr * atf_price
            lease_crew_cost = seats * RNG.uniform(950, 1400) * 4.33 * (weekly_frequency / 7)
            airport_handling = flights_month * RNG.uniform(18000, 32000)

            total_revenue = ticket_revenue + ancillary_revenue + udan_subsidy
            total_cost = fuel_cost + lease_crew_cost + airport_handling
            ebitda = total_revenue - total_cost
            ebitda_margin = ebitda / total_revenue if total_revenue > 0 else np.nan

            rows.append(dict(
                route_id=route_id, route=pair, month=month.strftime("%Y-%m"),
                seats=seats, weekly_frequency=weekly_frequency, block_hours=block_hours,
                is_udan_route=is_udan, atf_price_per_litre=round(atf_price, 2),
                load_factor=round(load_factor, 3), avg_fare=round(avg_fare, 0),
                passengers=round(pax_month, 0), ticket_revenue=round(ticket_revenue, 0),
                ancillary_revenue=round(ancillary_revenue, 0), udan_subsidy=round(udan_subsidy, 0),
                fuel_cost=round(fuel_cost, 0), lease_crew_cost=round(lease_crew_cost, 0),
                airport_handling_cost=round(airport_handling, 0),
                total_revenue=round(total_revenue, 0), total_cost=round(total_cost, 0),
                ebitda=round(ebitda, 0), ebitda_margin=round(ebitda_margin, 4),
            ))

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows -> {OUT_PATH}")
    return df


if __name__ == "__main__":
    generate()
