import subprocess
import sys

def run(mod: str):
    print(f"\n=== Running: {mod} ===")
    subprocess.check_call([sys.executable, "-m", mod])

def main():
    run("src.make_parquet")
    run("src.build_dataset")
    run("src.train_xgb_time_split")
    run("src.tune_thresholds")
    run("src.make_submission")
    run("src.report_plots")

if __name__ == "__main__":
    main()