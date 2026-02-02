import subprocess
import time
import os
import sys
import matplotlib.pyplot as plt

support_thresholds = [5, 10, 25, 50, 90]
tout_sec = 3600

#### LLM-assisted code (ChatGPT); all logic and correctness verified by the authors.
def run_algo(cmd):
    start = time.time()
    try:
        subprocess.run(cmd, timeout=tout_sec)
        return time.time() - start
    except subprocess.TimeoutExpired:
        return tout_sec
#### 

def main():
    apriori = sys.argv[1]
    fpgrowth = sys.argv[2]
    dataset = sys.argv[3]
    out_dir = sys.argv[4]

    os.makedirs(out_dir, exist_ok=True)

    ap_times = []
    fp_times = []

    for s in support_thresholds:
        ap_dir = os.path.join(out_dir, f"ap{s}")
        fp_dir = os.path.join(out_dir, f"fp{s}")

        os.makedirs(ap_dir, exist_ok=True)
        os.makedirs(fp_dir, exist_ok=True)

        ap_out = os.path.join(ap_dir, "out.txt")
        fp_out = os.path.join(fp_dir, "out.txt")

        ap_cmd = [
            apriori,
            "-s" + str(s),
            dataset,
            ap_out
        ]

        fp_cmd = [
            fpgrowth,
            "-s" + str(s),
            dataset,
            fp_out
        ]

        ap_times.append(run_algo(ap_cmd))
        fp_times.append(run_algo(fp_cmd))

    plt.figure()
    plt.plot(support_thresholds, ap_times, marker="o", label="Apriori")
    plt.plot(support_thresholds, fp_times, marker="s", label="FP-Tree")
    plt.xlabel("Minimum Support (%)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Empirical Comparison - Apriori vs FP-Tree")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "plot.png"))

if __name__ == "__main__":
    main()
