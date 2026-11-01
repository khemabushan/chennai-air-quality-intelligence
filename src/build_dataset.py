"""Raw CSVs -> data/processed/daily.csv (explicit calendar) and features.csv."""
import json

from src.config import DATA_PROCESSED, METRICS_DIR
from src.data_loader import load_raw, verify_raw_hashes
from src.data_validation import check_schema
from src.feature_engineering import build_feature_table
from src.preprocessing import to_daily_calendar


def main() -> None:
    hashes = verify_raw_hashes()
    if not all(hashes.values()):
        raise RuntimeError(f"raw CSV hash mismatch: {hashes}")
    raw = load_raw()
    problems = check_schema(raw)
    if problems:
        raise RuntimeError(problems)
    daily = to_daily_calendar(raw)
    feats = build_feature_table(daily)
    daily.to_csv(DATA_PROCESSED / "daily.csv", index=False)
    feats.to_csv(DATA_PROCESSED / "features.csv", index=False)
    counts = feats.groupby("split").size().to_dict()
    (METRICS_DIR / "dataset_counts.json").write_text(json.dumps(
        {"raw_rows": len(raw), "calendar_rows": len(daily), "eligible_origins": len(feats),
         "per_split": counts}, indent=2))
    print(f"raw={len(raw)} calendar={len(daily)} eligible={len(feats)} splits={counts}")


if __name__ == "__main__":
    main()
