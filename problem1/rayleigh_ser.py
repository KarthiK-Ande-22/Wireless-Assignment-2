"""Problem 1, Task 2: SER of BPSK and 4-QAM in Rayleigh fading, h ~ CN(0, 1).

h is drawn independently for every symbol and is known at the receiver
(coherent ML detection). The script also produces the numbers and the extra
figure used to discuss AWGN versus fading: the BPSK / 4-QAM gap, the extra SNR
that fading costs, and how many errors happen during a deep fade.

Run from the repository root:   python -m problem1.rayleigh_ser
Outputs:                        results/q1/rayleigh_ser.{csv,pdf}, results/q1/images/rayleigh_ser.png
                                results/q1/awgn_vs_rayleigh.pdf, results/q1/images/awgn_vs_rayleigh.png
                                results/q1/rayleigh_summary.txt
"""
from pathlib import Path

import numpy as np

from common.analysis import agreement_lines, fitted_slope, snr_at_ser, theory_snr_at_ser
from common.channel import rayleigh_channel
from common.constellation import bpsk, qam4
from common.detector import ml_detect
from common.plotting import PALETTE, GapArrow, SimCurve, TheoryCurve, plot_ser, save_results
from common.simulation import db_to_linear, simulate_ser
from common.theory import (ser_bpsk_awgn, ser_bpsk_rayleigh, ser_qam4_awgn,
                           ser_qam4_rayleigh, ser_qam4_rayleigh_exact)

SEED = 2026
N_SYMBOLS = 10**6
SNR_DB = np.arange(0, 41, 2)   # {0, 2, ..., 40} dB
GAP_TARGETS = (1e-2, 1e-3, 1e-4)   # SER levels at which gaps are read
DEEP_FADE_SNR_DB = (10, 20, 30, 40)   # SNR points of the deep-fade measurement
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "q1"


def deep_fade_shares(constellation, snr_db, rng):
    """Share of symbols sent in a deep fade (|h|^2 SNR < 1) and share of the errors made in one.

    Runs Algorithm 1 again but keeps h for every symbol, so each error can be
    traced back to the channel gain it was received with.
    """
    rows = []
    for snr in db_to_linear(snr_db):
        i = rng.integers(len(constellation), size=N_SYMBOLS)          # ground truth
        h, y = rayleigh_channel(constellation[i], snr, rng)            # channel
        error = ml_detect(y, constellation, h) != i                    # detector, error indicator
        deep = np.abs(h) ** 2 * snr < 1                                # received SNR below 0 dB
        rows.append((deep.mean(), error.sum(), deep[error].mean()))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    snr = db_to_linear(SNR_DB)

    # 1. Simulation
    ser_bpsk, nerr_bpsk = simulate_ser(bpsk(), SNR_DB, rayleigh_channel, N_SYMBOLS, rng)
    ser_qam4, nerr_qam4 = simulate_ser(qam4(), SNR_DB, rayleigh_channel, N_SYMBOLS, rng)
    sims = [SimCurve("BPSK", ser_bpsk, nerr_bpsk, PALETTE[0], "o"),
            SimCurve("4-QAM", ser_qam4, nerr_qam4, PALETTE[1], "s")]

    # 2. Theory (table expressions; the exact 4-QAM curve explains the gap)
    theories = [TheoryCurve("BPSK theory", ser_bpsk_rayleigh(snr), PALETTE[0]),
                TheoryCurve("4-QAM theory (approx.)", ser_qam4_rayleigh(snr), PALETTE[1]),
                TheoryCurve("4-QAM theory (exact)", ser_qam4_rayleigh_exact(snr),
                            PALETTE[1], linestyle=":")]

    # 3. SNR needed at fixed SER: BPSK and 4-QAM in fading, BPSK and 4-QAM in AWGN (theory)
    fading = {t: (snr_at_ser(SNR_DB, ser_bpsk, nerr_bpsk, t), snr_at_ser(SNR_DB, ser_qam4, nerr_qam4, t),
                  theory_snr_at_ser(ser_bpsk_rayleigh, t), theory_snr_at_ser(ser_qam4_rayleigh_exact, t))
              for t in GAP_TARGETS}
    awgn = {t: (theory_snr_at_ser(ser_bpsk_awgn, t), theory_snr_at_ser(ser_qam4_awgn, t)) for t in GAP_TARGETS}
    arrows = [GapArrow(fading[t][0], fading[t][1], t, f"{fading[t][1] - fading[t][0]:.1f} dB",
                       label_side="left")                              # measured BPSK -> 4-QAM gap
              for t in (1e-2, 1e-3)]

    # 4. Plot and save the Task 2 figure
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_results(OUT_DIR / "rayleigh_ser", SNR_DB, sims, theories,
                 title="Rayleigh fading, coherent ML detection, N = $10^6$ symbols per SNR point",
                 arrows=arrows)

    # 5. Extra figure: AWGN (theory) against fading, with the deep-fade probability
    fine_db = np.arange(0, 40.01, 0.1)                                 # smooth AWGN waterfall
    fine = db_to_linear(fine_db)
    comparison = [
        TheoryCurve("BPSK, AWGN (theory)", ser_bpsk_awgn(fine), PALETTE[0], "--", snr_db=fine_db),
        TheoryCurve("4-QAM, AWGN (theory)", ser_qam4_awgn(fine), PALETTE[1], "--", snr_db=fine_db),
        TheoryCurve("BPSK, fading (theory)", ser_bpsk_rayleigh(snr), PALETTE[0]),
        TheoryCurve("4-QAM, fading (exact)", ser_qam4_rayleigh_exact(snr), PALETTE[1]),
        TheoryCurve("deep-fade probability", 1 - np.exp(-1 / fine), "black", ":", snr_db=fine_db)]
    fade_arrows = [GapArrow(awgn[t][0], fading[t][0], t, f"BPSK: {fading[t][0] - awgn[t][0]:.1f} dB")
                   for t in (1e-3, 1e-4)]                              # AWGN theory -> fading simulation
    plot_ser(OUT_DIR / "awgn_vs_rayleigh", SNR_DB,
             [SimCurve("BPSK", ser_bpsk, nerr_bpsk, PALETTE[0], "o", label="BPSK, fading (sim.)"),
              SimCurve("4-QAM", ser_qam4, nerr_qam4, PALETTE[1], "s", label="4-QAM, fading (sim.)")],
             comparison, title="AWGN vs Rayleigh fading, N = $10^6$ symbols per SNR point",
             legend_loc="upper right", arrows=fade_arrows)
    print(f"Saved {OUT_DIR / 'awgn_vs_rayleigh'}.pdf and images/awgn_vs_rayleigh.png\n")

    # 6. Numbers quoted in the discussion
    approx, exact = theories[1].ser, theories[2].ser
    lines = agreement_lines([("BPSK vs BPSK theory", ser_bpsk, nerr_bpsk, theories[0].ser),
                             ("4-QAM vs 4-QAM theory (approx.)", ser_qam4, nerr_qam4, approx),
                             ("4-QAM vs 4-QAM theory (exact)", ser_qam4, nerr_qam4, exact)],
                            N_SYMBOLS)
    lines += ["", "4-QAM approximation / exact - 1 (overestimate of the table expression)",
              "  " + ", ".join(f"{s} dB: {r:+.1%}" for s, r in zip(SNR_DB[::5], approx[::5] / exact[::5] - 1))]
    lines += ["", "High-SNR slope (diversity order): least-squares fit over 30-40 dB"]
    lines += [f"{c.name:>8}: {fitted_slope(SNR_DB, c.ser, c.n_errors, 30):.2f}" for c in sims]

    high = 1e8                                                          # 80 dB: high-SNR limit
    lines += ["", "SNR (dB) needed in fading: simulation (exact theory in brackets)",
              f"{'SER':>8}{'BPSK':>16}{'4-QAM':>16}{'gap 4-QAM - BPSK':>20}"]
    for t, (b, q, b_th, q_th) in fading.items():
        lines.append(f"{t:>8.0e}{f'{b:.2f} ({b_th:.2f})':>16}{f'{q:.2f} ({q_th:.2f})':>16}"
                     f"{f'{q - b:.2f} ({q_th - b_th:.2f})':>20}")
    lines.append(f"High-SNR forms: BPSK Pe -> {ser_bpsk_rayleigh(high) * high:.4f} / SNR, "
                 f"4-QAM Pe -> {ser_qam4_rayleigh_exact(high) * high:.4f} / SNR; "
                 f"gap limit {10 * np.log10(ser_qam4_rayleigh_exact(high) / ser_bpsk_rayleigh(high)):.2f} dB")

    lines += ["", "Price of fading: SNR (dB) for a target SER, AWGN (theory) vs Rayleigh (simulation)",
              f"{'scheme':>8}{'SER':>8}{'AWGN':>10}{'Rayleigh':>12}{'extra SNR':>12}"]
    for name, col in (("BPSK", 0), ("4-QAM", 1)):
        for t in (1e-3, 1e-4):
            lines.append(f"{name:>8}{t:>8.0e}{awgn[t][col]:>10.2f}{fading[t][col]:>12.2f}"
                         f"{fading[t][col] - awgn[t][col]:>12.2f}")

    lines += ["", "Deep fades (|h|^2 SNR < 1), BPSK in Rayleigh fading, N = 10^6 symbols per point",
              f"{'SNR_dB':>8}{'symbols in deep fade':>22}{'errors':>9}{'errors in deep fade':>21}"]
    for s, (share_symbols, n_err, share_errors) in zip(DEEP_FADE_SNR_DB,
                                                       deep_fade_shares(bpsk(), DEEP_FADE_SNR_DB, rng)):
        lines.append(f"{s:>8}{share_symbols:>22.4%}{n_err:>9}{share_errors:>21.1%}")
    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "rayleigh_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
