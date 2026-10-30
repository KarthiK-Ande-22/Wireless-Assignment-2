"""Compare every plotted point of Problems 1-3 with its exact theory.

For each point, z = (n_err - N p) / sqrt(N p (1 - p)) should behave like a standard
normal value if the simulation is correct. This script reads the CSV files of the
four problem scripts, so run them first.

Run:     python3 -m overall_check
Output:  results/overall_check.txt
"""
from math import erf, sqrt
from pathlib import Path

import numpy as np

from common.plotting import MIN_ERRORS

N_SYMBOLS = 10**6
RESULTS = Path(__file__).resolve().parent / "results"

# (problem, curve name, CSV file, column of the simulation, column of the exact theory)
CURVES = [
    (1, "BPSK, AWGN", "q1/awgn_ser.csv", "BPSK", "BPSK theory"),
    (1, "4-QAM, AWGN", "q1/awgn_ser.csv", "4-QAM", "4-QAM theory"),
    (1, "BPSK, fading", "q1/rayleigh_ser.csv", "BPSK", "BPSK theory"),
    (1, "4-QAM, fading", "q1/rayleigh_ser.csv", "4-QAM", "4-QAM theory (exact)"),
    (2, "MRC, L = 1 (= SC, L = 1)", "q2/receive_diversity_theory.csv", "MRC L=1", "MRC L=1 exact theory"),
    (2, "MRC, L = 2", "q2/receive_diversity_theory.csv", "MRC L=2", "MRC L=2 exact theory"),
    (2, "MRC, L = 4", "q2/receive_diversity_theory.csv", "MRC L=4", "MRC L=4 exact theory"),
    (2, "SC, L = 2", "q2/receive_diversity_theory.csv", "SC L=2", "SC L=2 exact theory"),
    (2, "SC, L = 4", "q2/receive_diversity_theory.csv", "SC L=4", "SC L=4 exact theory"),
    (3, "Beamforming", "q3/transmit_diversity_theory.csv", "Beamforming", "Beamforming exact theory"),
    (3, "Alamouti", "q3/transmit_diversity_theory.csv", "Alamouti", "Alamouti exact theory"),
    (3, "MRC, L = 2", "q3/transmit_diversity_theory.csv", "MRC L=2", "MRC L=2 exact theory"),
]


def read_csv(name):
    """Read a CSV file of results/ as {column name: values}."""
    path = RESULTS / name
    header = path.read_text().splitlines()[0].split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return {column: data[:, k] for k, column in enumerate(header)}


def main():
    tables = {name: read_csv(name) for name in {c[2] for c in CURVES}}

    # MRC L=1, SC L=1 and the Problem 3 single antenna use the same random numbers: count them once
    q2, q3 = read_csv("q2/receive_diversity.csv"), read_csv("q3/transmit_diversity.csv")
    assert np.array_equal(q2["SC L=1 n_err"], q2["MRC L=1 n_err"])
    assert np.array_equal(q3["Single antenna n_err"], q2["MRC L=1 n_err"])

    lines = ["Overall check: every plotted point (n_err >= 10) against its exact theory,",
             "z = (n_err - N p) / sqrt(N p (1 - p)), N = 10^6",
             "Counted once (identical error counts at every SNR): MRC L=1 and SC L=1 of Problem 2,",
             "and the single-antenna curve of Problem 3.", "",
             f"{'problem':>8}  {'curve':<26}{'points':>7}{'max |z|':>9}"]
    z_all = []
    for problem, label, name, prefix, theory in CURVES:
        n_err, p = tables[name][f"{prefix} n_err"], tables[name][theory]
        plotted = n_err >= MIN_ERRORS   # the same points as in the figures
        z = (n_err[plotted] - N_SYMBOLS * p[plotted]) / np.sqrt(N_SYMBOLS * p[plotted] * (1 - p[plotted]))
        z_all.append(z)
        lines.append(f"{problem:>8}  {label:<26}{plotted.sum():>7}{np.abs(z).max():>9.2f}")
    z = np.concatenate(z_all)
    n = z.size

    def within(k):
        return erf(k / sqrt(2))   # probability that |N(0, 1)| < k

    z_max = np.abs(z).max()
    chance = 1 - within(z_max) ** n   # chance that the largest of n values is at least z_max
    lines += ["", f"Statistics over all {n} independent points",
              f"{'':<28}{'measured':>10}{'expected if unbiased':>24}"]
    lines += [f"{f'points with |z| < {k}':<28}{np.mean(np.abs(z) < k):>10.1%}{within(k):>24.1%}"
              for k in (1, 2, 3)]
    lines += [f"{'largest |z|':<28}{z_max:>10.2f}{f'(reached by chance in {chance:.0%} of runs)':>40}",
              f"{'mean of z (bias)':<28}{z.mean():>+10.3f}{f'0 +- {1 / sqrt(n):.2f}':>24}",
              f"{'mean of z^2':<28}{np.mean(z ** 2):>10.2f}{f'1 +- {sqrt(2 / n):.2f}':>24}"]
    summary = "\n".join(lines) + "\n"
    (RESULTS / "overall_check.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
