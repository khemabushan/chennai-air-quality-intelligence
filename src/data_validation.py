"""Data-quality checks. These functions only report; they never alter measurements."""
import numpy as np
import pandas as pd

from src.config import CALENDAR_END, CALENDAR_START, POLLUTANTS, RAW_COLUMNS, STATIONS


def check_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of schema problems (empty list means OK)."""
    problems = []
    for col in ["station", "date", *POLLUTANTS]:
        if col not in df.columns:
            problems.append(f"missing column {col}")
    if "date" in df and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        problems.append("date is not datetime")
    for col in POLLUTANTS:
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col} is not numeric")
    unknown = set(df.get("station", pd.Series(dtype=str)).unique()) - set(STATIONS)
    if unknown:
        problems.append(f"unknown stations {sorted(unknown)}")
    return problems


def station_summary(df: pd.DataFrame) -> pd.DataFrame:
    full_days = len(pd.date_range(CALENDAR_START, CALENDAR_END, freq="D"))
    rows = []
    for st, g in df.groupby("station"):
        dates = g["date"].sort_values()
        gaps = dates.diff().dt.days.dropna()
        rows.append({
            "station": st,
            "rows": len(g),
            "first_date": dates.min().date(),
            "last_date": dates.max().date(),
            "calendar_days": full_days,
            "missing_dates": full_days - dates.nunique(),
            "duplicate_dates": int(dates.duplicated().sum()),
            "longest_gap_days": int(gaps.max() - 1) if len(gaps) else 0,
            "gaps_over_7_days": int((gaps > 8).sum()),
        })
    return pd.DataFrame(rows)


def longest_gaps(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    out = []
    for st, g in df.groupby("station"):
        d = g["date"].sort_values().reset_index(drop=True)
        diff = d.diff().dt.days
        for i in np.where(diff > 1)[0]:
            out.append({"station": st, "gap_start": (d[i - 1] + pd.Timedelta(days=1)).date(),
                        "gap_end": (d[i] - pd.Timedelta(days=1)).date(), "missing_days": int(diff[i] - 1)})
    return pd.DataFrame(out).sort_values("missing_days", ascending=False).head(top).reset_index(drop=True)


def value_checks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in POLLUTANTS:
        s = df[col]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        rows.append({
            "pollutant": col,
            "missing": int(s.isna().sum()),
            "negative": int((s < 0).sum()),
            "zero": int((s == 0).sum()),
            "min": s.min(), "median": s.median(), "mean": round(s.mean(), 2),
            "p99": s.quantile(0.99), "max": s.max(),
            "iqr_outliers_(>Q3+3IQR)": int((s > q3 + 3 * iqr).sum()),
            "most_common_value": s.mode().iloc[0],
            "most_common_share_%": round(100 * (s == s.mode().iloc[0]).mean(), 2),
        })
    return pd.DataFrame(rows)


def pm25_gt_pm10(df: pd.DataFrame) -> pd.DataFrame:
    flag = df["PM2.5"] > df["PM10"]
    return (df.assign(flag=flag).groupby("station")["flag"]
            .agg(rows="size", pm25_gt_pm10="sum")
            .assign(share_pct=lambda t: (100 * t.pm25_gt_pm10 / t.rows).round(1))
            .reset_index())


def constant_runs(df: pd.DataFrame, col: str = "PM2.5", min_len: int = 3) -> pd.DataFrame:
    """Runs of identical consecutive (by record order) readings of length >= min_len."""
    out = []
    for st, g in df.sort_values("date").groupby("station"):
        v = g[col].to_numpy()
        d = g["date"].to_numpy()
        start = 0
        for i in range(1, len(v) + 1):
            if i == len(v) or v[i] != v[start]:
                if i - start >= min_len:
                    out.append({"station": st, "pollutant": col, "value": v[start],
                                "start": pd.Timestamp(d[start]).date(),
                                "end": pd.Timestamp(d[i - 1]).date(), "length": i - start})
                start = i
    if not out:
        return pd.DataFrame(columns=["station", "pollutant", "value", "start", "end", "length"])
    return pd.DataFrame(out).sort_values("length", ascending=False).reset_index(drop=True)
