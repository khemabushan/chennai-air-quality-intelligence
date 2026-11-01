import numpy as np
import pandas as pd
import pytest

from src.config import STATIONS
from src.data_loader import load_raw
from src.feature_engineering import FEATURES, TARGET_COL, build_feature_table
from src.preprocessing import to_daily_calendar


@pytest.fixture(scope="module")
def daily():
    return to_daily_calendar(load_raw())


@pytest.fixture(scope="module")
def feats(daily):
    return build_feature_table(daily)


def _value(daily, station, date):
    v = daily[(daily.station == station) & (daily.date == pd.Timestamp(date))]["PM2.5"]
    return v.iloc[0] if len(v) else np.nan  # dates before 2022-01-01 are unknown


def test_target_is_next_calendar_day(daily, feats):
    assert (feats["target_date"] - feats["origin_date"]).eq(pd.Timedelta(days=1)).all()
    sample = feats.sample(200, random_state=0)
    for r in sample.itertuples():
        assert getattr(r, TARGET_COL) == _value(daily, r.station, r.target_date)
        assert r.pm25_t == _value(daily, r.station, r.origin_date)
        lag7 = _value(daily, r.station, r.origin_date - pd.Timedelta(days=7))
        assert (np.isnan(lag7) and np.isnan(r.pm25_lag7)) or lag7 == r.pm25_lag7


def test_eligibility_requires_observed_origin_and_label(feats):
    assert feats["pm25_t"].notna().all()
    assert feats[TARGET_COL].notna().all()


def test_no_future_leakage(daily):
    """Changing every value after an origin must not change that origin's features."""
    st, origin = "Velachery", pd.Timestamp("2024-03-15")
    base = build_feature_table(daily, drop_ineligible=False)
    tampered = daily.copy()
    future = (tampered.station == st) & (tampered.date > origin)
    tampered.loc[future, ["PM2.5", "PM10", "NO2", "SO2", "O3"]] = 9999.0
    alt = build_feature_table(tampered, drop_ineligible=False)
    key = (base.station == st) & (base.origin_date == origin)
    pd.testing.assert_series_equal(base.loc[key, FEATURES].iloc[0], alt.loc[key, FEATURES].iloc[0])
    # ...while the label does change, proving the label is the only future-dependent column
    assert alt.loc[key, TARGET_COL].iloc[0] == 9999.0


def test_stations_do_not_mix(daily):
    """Tampering with one station leaves every other station's features untouched."""
    base = build_feature_table(daily)
    tampered = daily.copy()
    tampered.loc[tampered.station == "Manali", "PM2.5"] += 1000
    alt = build_feature_table(tampered)
    others = base.station != "Manali"
    pd.testing.assert_frame_equal(base[others].reset_index(drop=True), alt[alt.station != "Manali"].reset_index(drop=True))


def test_split_assigned_by_target_date(feats):
    assert feats.loc[feats.split == "train", "target_date"].max() <= pd.Timestamp("2023-12-31")
    v = feats[feats.split == "val"]["target_date"]
    assert v.min() >= pd.Timestamp("2024-01-01") and v.max() <= pd.Timestamp("2024-06-30")
    assert feats.loc[feats.split == "test", "target_date"].min() >= pd.Timestamp("2024-07-01")


def test_station_one_hot(feats):
    oh = feats[[f"station_{s}" for s in STATIONS]]
    assert oh.sum(axis=1).eq(1).all()
