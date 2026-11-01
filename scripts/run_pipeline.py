"""Run the full workflow end-to-end: audit -> data -> baselines -> ML -> LSTM -> evaluation -> XAI.

Usage (from the repository root):  python scripts/run_pipeline.py [--skip-lstm]
"""
import subprocess
import sys
import time

STEPS = [
    "src.audit",
    "src.build_dataset",
    "src.train_baselines",
    "src.train_ml",
    "src.train_lstm",
    "src.evaluate",
    "src.plots",
    "src.explainability",
]

if __name__ == "__main__":
    skip_lstm = "--skip-lstm" in sys.argv
    for step in STEPS:
        if skip_lstm and step == "src.train_lstm":
            continue
        t0 = time.time()
        print(f"\n>>> python -m {step}", flush=True)
        subprocess.run([sys.executable, "-m", step], check=True)
        print(f"<<< {step} done in {time.time() - t0:.1f}s", flush=True)
