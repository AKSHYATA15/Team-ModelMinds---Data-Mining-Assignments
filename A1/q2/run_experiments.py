#LLM generated code

import os
import time
import argparse
import subprocess
import matplotlib.pyplot as plt

MIN_SUPS = [5, 10, 25, 50, 95]
TIMEOUT = 20 * 60  # 20 minutes timeout 


def run_command(cmd, outfile, timeout=TIMEOUT):
    print("--------------------------------------------------")
    print("Running command:")
    print(" ".join(cmd))

    start = time.time()
    status = "OK"

    try:
        with open(outfile, "w") as f:
            subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.DEVNULL,
                timeout=timeout
            )
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"

    elapsed = min(time.time() - start, timeout)

    print(f"Status: {status}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    return elapsed, status


def main(args):
    times = {"gspan": [], "fsg": [], "gaston": []}

    gspan_input = "processed/yeast_gspan.txt"
    fsg_input = "processed/yeast_fsg.txt"
    gaston_input = "processed/yeast_gaston.txt"

    for sup in MIN_SUPS:
        print(f"\n================= MIN SUPPORT = {sup}% =================")

        # -------- gSpan --------
        gspan_dir = os.path.join(args.outdir, f"gspan{sup}")
        os.makedirs(gspan_dir, exist_ok=True)
        gspan_outfile = os.path.join(gspan_dir, "patterns.txt")

        t, _ = run_command(
            [args.gspan, "-f", gspan_input, "-s", str(sup), "-o"],
            gspan_outfile
        )
        times["gspan"].append(t)

        # -------- FSG --------
        fsg_dir = os.path.join(args.outdir, f"fsg{sup}")
        os.makedirs(fsg_dir, exist_ok=True)
        fsg_outfile = os.path.join(fsg_dir, "patterns.fp")

        t, _ = run_command(
            [args.fsg, "-s", str(sup), fsg_input],
            fsg_outfile
        )
        times["fsg"].append(t)

        # -------- Gaston --------
        gaston_dir = os.path.join(args.outdir, f"gaston{sup}")
        os.makedirs(gaston_dir, exist_ok=True)
        gaston_outfile = os.path.join(gaston_dir, "patterns.txt")

        t, _ = run_command(
            [args.gaston, str(sup), gaston_input],
            gaston_outfile
        )
        times["gaston"].append(t)

    # -------- Plot --------
    plt.figure(figsize=(8, 6))
    plt.plot(MIN_SUPS, times["gspan"], marker="o", label="gSpan")
    plt.plot(MIN_SUPS, times["fsg"], marker="s", label="FSG")
    plt.plot(MIN_SUPS, times["gaston"], marker="^", label="Gaston")

    plt.xlabel("Minimum Support (%)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Minimum Support (Yeast Dataset)")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(args.outdir, "plot.png"))
    plt.close()

    print("\n================= ALL DONE =================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gspan", required=True)
    parser.add_argument("--fsg", required=True)
    parser.add_argument("--gaston", required=True)
    parser.add_argument("--dataset", required=True)  # kept for interface
    parser.add_argument("--outdir", required=True)
    main(parser.parse_args())
