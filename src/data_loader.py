"""Load the original station CSVs (read-only) into one long-format DataFrame."""
import hashlib
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, RAW_COLUMNS, RAW_SHA256, STATIONS


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_raw_hashes(raw_dir: Path = DATA_RAW) -> dict:
    """Return {filename: matches_expected_hash}."""
    return {name: sha256(raw_dir / name) == expected for name, expected in RAW_SHA256.items()}


def load_station_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df = df[RAW_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="raise")
    df["station"] = df["MONITORING STATION"].str.strip().str.title()
    return df


def load_raw(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    """All six stations, sorted by station then date. Values are not modified."""
    frames = [load_station_csv(raw_dir / f"{s}.csv") for s in STATIONS]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop(columns=["MONITORING STATION"]).rename(columns={"Date": "date"})
    df = df[["station", "date", "PM2.5", "PM10", "NO2", "SO2", "O3"]]
    return df.sort_values(["station", "date"]).reset_index(drop=True)
