"""Central configuration: paths, dataset constants, split boundaries and seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
REPORTS_DIR = ROOT / "reports"

STATIONS = ["Alandur", "Kodungaiyur", "Manali", "Perungudi", "Royapuram", "Velachery"]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "O3"]
TARGET = "PM2.5"
RAW_COLUMNS = ["Date", "MONITORING STATION", "PM2.5", "PM10", "NO2", "SO2", "O3"]

CALENDAR_START = "2022-01-01"
CALENDAR_END = "2024-12-31"

# Splits are assigned by TARGET date (the day being forecast).
TRAIN_END = "2023-12-31"
VAL_END = "2024-06-30"
TEST_END = "2024-12-31"

SEED = 42

# SHA-256 of the original CSVs extracted from the Mendeley archive (DOI 10.17632/br5dyxm7c6.1).
RAW_SHA256 = {
    "Alandur.csv": "395d01123bf56a8954398a58f4088fe495d7544157fa9ba7b5514762faca8bed",
    "Kodungaiyur.csv": "72a8a4de01350f485a1f824d1ea7e8c834da3a5cbb2137458751804f7cc77d74",
    "Manali.csv": "fc2368117a662a16852140f9bbc34f6bb018491f586cf0710ae519df686cc7ef",
    "Perungudi.csv": "06b4adecb913b10310cc6b7b3c85ed691702a7b64540213aa46bc3291a5177e1",
    "Royapuram.csv": "211de001274c25874045ac995037481e92df28f39dcc2a172797e3bf8cae52bc",
    "Velachery.csv": "12ac75324c743048c0c080587252d34ac5a0f60942d71c9a2789b89166bbe7db",
}

for _d in (DATA_PROCESSED, MODELS_DIR, METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
