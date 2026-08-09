import pandas as pd
import matplotlib.pyplot as plt

def main():
    scored = pd.read_csv("reports/test_scored.csv")

    plt.figure()
    scored["risk_score"].hist(bins=50)
    plt.xlabel("Risk score (0-100)")
    plt.ylabel("Count")
    plt.title("Test Risk Score Distribution")
    plt.tight_layout()
    plt.savefig("reports/risk_score_hist.png", dpi=200)

    plt.figure()
    scored["risk_band"].value_counts().plot(kind="bar")
    plt.xlabel("Risk band")
    plt.ylabel("Count")
    plt.title("Test Risk Band Counts")
    plt.tight_layout()
    plt.savefig("reports/risk_band_counts.png", dpi=200)

    print("Saved reports/risk_score_hist.png")
    print("Saved reports/risk_band_counts.png")

if __name__ == "__main__":
    main()