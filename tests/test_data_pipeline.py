"""
Lightweight sanity tests for the data-generation and analysis pipeline.
Run with:  pytest tests/ -v

These aren't exhaustive unit tests of every function -- they exist to
catch the class of bug that actually broke this project during
development (e.g. a revenue/cost unit mismatch, a pandas NA-string
collision) before they silently produce nonsense numbers in a report.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_generation.generate_airline_data import generate as gen_airline
from data_generation.generate_ott_data import generate as gen_ott
from data_generation.generate_telemedicine_data import generate as gen_telemedicine


class TestAirlineData:
    @pytest.fixture(scope="class")
    def df(self):
        return gen_airline()

    def test_no_nulls_in_key_columns(self, df):
        key_cols = ["route_id", "month", "total_revenue", "total_cost", "ebitda"]
        assert df[key_cols].isna().sum().sum() == 0

    def test_revenue_and_cost_same_order_of_magnitude(self, df):
        # Regression test for the revenue/cost unit-mismatch bug found during
        # development (revenue was computed per-flight, cost per-month).
        ratio = df["total_revenue"].mean() / df["total_cost"].mean()
        assert 0.3 < ratio < 3.0, f"revenue/cost ratio {ratio:.2f} suggests a unit mismatch"

    def test_load_factor_in_valid_range(self, df):
        assert df["load_factor"].between(0, 1).all()

    def test_post_shock_atf_price_higher_than_pre_shock(self, df):
        df["month"] = pd.to_datetime(df["month"])
        pre = df[df["month"] < "2024-10"]["atf_price_per_litre"].mean()
        post = df[df["month"] >= "2024-10"]["atf_price_per_litre"].mean()
        assert post > pre


class TestOTTData:
    @pytest.fixture(scope="class")
    def df(self):
        return gen_ott()

    def test_dropoff_rate_bounded(self, df):
        assert df["dropoff_rate"].between(0, 1).all()

    def test_dropoff_flag_is_binary(self, df):
        assert set(df["dropoff_flag"].unique()) <= {0, 1}

    def test_mid_season_slump_present(self, df):
        # Episodes 4-5 should have higher avg drop-off than episodes 1-2
        ep_avg = df.groupby("episode_number")["dropoff_rate"].mean()
        assert ep_avg[[4, 5]].mean() > ep_avg[[1, 2]].mean()

    def test_all_series_have_full_season(self, df):
        counts = df.groupby("series_id").size()
        assert (counts == 8).all()


class TestTelemedicineData:
    @pytest.fixture(scope="class")
    def df(self):
        return gen_telemedicine()

    def test_no_show_is_binary(self, df):
        assert set(df["no_show"].unique()) <= {0, 1}

    def test_reminder_channel_no_pandas_na_collision(self, df):
        # Regression test: the literal string "None" is silently parsed as
        # NaN by pandas on CSV read-back. Every row must have a real,
        # non-null reminder_channel value.
        assert df["reminder_channel"].isna().sum() == 0

    def test_first_time_users_have_zero_prior_consults(self, df):
        ft = df[df["first_time_user"]]
        assert (ft["prior_completed_consults"] == 0).all()

    def test_noshow_rate_is_realistic(self, df):
        rate = df["no_show"].mean()
        assert 0.05 < rate < 0.35, f"no-show rate {rate:.1%} is outside a realistic range"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
