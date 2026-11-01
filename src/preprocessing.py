"""Time-series-safe preprocessing.

Each station is reindexed onto a complete daily calendar so missing days become explicit
NaN rows (with an ``observed`` flag). No value is imputed here: no forward/backward filling,
no interpolation. Downstream feature code decides causally how to treat gaps. The only change to
recorded values is the documented constant-run QC mask below.
"""
import pandas as pd

from src.config import (CALENDAR_END, CALENDAR_START, POLLUTANTS, STATIONS, TEST_END,
                        TRAIN_END, VAL_END)

QC_CONSTANT_RUN_DAYS = 14


def to_daily_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.duplicated(["station", "date"]).any():
        raise ValueError("duplicate (station, date) rows in raw data")
    calendar = pd.date_range(CALENDAR_START, CALENDAR_END, freq="D")
    frames = []
    for st in STATIONS:
        g = raw[raw["station"] == st].set_index("date")[POLLUTANTS]
        g = g.reindex(calendar)
        g.index.name = "date"
        g = g.reset_index()
        g.insert(0, "station", st)
        g["observed"] = g["PM2.5"].notna()
        frames.append(g)
    return mask_constant_pm25_runs(pd.concat(frames, ignore_index=True))


def mask_constant_pm25_runs(daily: pd.DataFrame, min_len: int = QC_CONSTANT_RUN_DAYS) -> pd.DataFrame:
    """Documented QC rule: PM2.5 identical on >= min_len consecutive calendar days is treated as a
    filler value, not a measurement, and set to NaN (flag ``pm25_qc_masked``). ``observed`` still
    records what the source file contains. Raw CSVs are never modified."""
    daily = daily.copy()
    daily["pm25_qc_masked"] = False
    for st, g in daily.groupby("station"):
        v = g["PM2.5"]
        run_id = (v != v.shift()).cumsum()
        run_len = v.groupby(run_id).transform("size")
        bad = v.notna() & (run_len >= min_len)
        daily.loc[g.index[bad.to_numpy()], "pm25_qc_masked"] = True
    daily.loc[daily["pm25_qc_masked"], "PM2.5"] = float("nan")
    return daily


def assign_split(target_date: pd.Series) -> pd.Series:
    """Chronological split label, assigned by the date being forecast."""
    split = pd.Series("test", index=target_date.index)
    split[target_date <= pd.Timestamp(VAL_END)] = "val"
    split[target_date <= pd.Timestamp(TRAIN_END)] = "train"
    split[target_date > pd.Timestamp(TEST_END)] = "out_of_range"
    return split
