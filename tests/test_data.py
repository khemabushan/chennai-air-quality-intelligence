import pandas as pd

from src.config import POLLUTANTS, STATIONS
from src.data_loader import load_raw, verify_raw_hashes
from src.data_validation import check_schema
from src.preprocessing import assign_split, to_daily_calendar


def test_raw_files_unchanged():
    assert all(verify_raw_hashes().values())


def test_load_raw_schema_and_dates():
    raw = load_raw()
    assert check_schema(raw) == []
    assert pd.api.types.is_datetime64_any_dtype(raw["date"])
    assert sorted(raw["station"].unique()) == STATIONS
    assert len(raw) == 6184
    assert raw["date"].min() == pd.Timestamp("2022-01-01")
    assert raw["date"].max() == pd.Timestamp("2024-12-31")
    assert not raw.duplicated(["station", "date"]).any()


def test_daily_calendar_exposes_missing_dates_without_imputation():
    raw = load_raw()
    daily = to_daily_calendar(raw)
    assert len(daily) == 6 * 1096
    assert daily.groupby("station").size().eq(1096).all()
    # observed rows equal raw rows and carry identical values
    assert daily["observed"].sum() == len(raw)
    merged = daily[daily.observed].merge(raw, on=["station", "date"], suffixes=("", "_raw"))
    for c in POLLUTANTS:
        same = merged[c] == merged[f"{c}_raw"]
        if c == "PM2.5":  # only the documented QC-masked filler run may differ
            assert (same | merged["pm25_qc_masked"]).all()
        else:
            assert same.all()
    assert daily.loc[~daily.observed, "PM2.5"].isna().all()


def test_qc_mask_targets_only_long_constant_runs():
    daily = to_daily_calendar(load_raw())
    masked = daily[daily.pm25_qc_masked]
    assert set(masked.station) == {"Kodungaiyur"}
    assert len(masked) == 126
    assert masked.date.max() == pd.Timestamp("2022-05-10")


def test_station_separation():
    daily = to_daily_calendar(load_raw())
    for st, g in daily.groupby("station"):
        assert g["date"].is_monotonic_increasing
        assert g["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()


def test_chronological_split_boundaries():
    d = pd.Series(pd.to_datetime(["2023-12-31", "2024-01-01", "2024-06-30", "2024-07-01", "2024-12-31"]))
    assert assign_split(d).tolist() == ["train", "val", "val", "test", "test"]
