"""Problem 1, Task 1: SER of BPSK and 4-QAM in the AWGN channel (h = 1).

Run from the repository root:   python -m problem1.awgn_ser
Outputs:                        results/q1/awgn_ser.{csv,pdf}, results/q1/images/awgn_ser.png
                                results/q1/awgn_summary.txt
"""
from pathlib import Path

import numpy as np

from common.analysis import agreement_lines, snr_at_ser, theory_snr_at_ser
from common.channel import awgn_channel
from common.constellation import bpsk, qam4
from common.plotting import PALETTE, GapArrow, SimCurve, TheoryCurve, save_results
from common.simulation import db_to_linear, simulate_ser
from common.theory import ser_bpsk_awgn, ser_qam4_awgn

SEED = 2026
N_SYMBOLS = 10**6
SNR_DB = np.arange(0, 13, 1)   # {0, 1, ..., 12} dB
GAP_TARGETS = (1e-2, 1e-3, 1e-4)   # SER levels at which the BPSK / 4-QAM gap is read
ARROW_TARGETS = (1e-2, 1e-4)       # gaps marked with arrows in the figure
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "q1"


def main():
    rng = np.random.default_rng(SEED)
    snr = db_to_linear(SNR_DB)

    # 1. Simulation
    ser_bpsk, nerr_bpsk = simulate_ser(bpsk(), SNR_DB, awgn_channel, N_SYMBOLS, rng)
    ser_qam4, nerr_qam4 = simulate_ser(qam4(), SNR_DB, awgn_channel, N_SYMBOLS, rng)
    sims = [SimCurve("BPSK", ser_bpsk, nerr_bpsk, PALETTE[0], "o"),
            SimCurve("4-QAM", ser_qam4, nerr_qam4, PALETTE[1], "s")]

    # 2. Theory
    theories = [TheoryCurve("BPSK theory", ser_bpsk_awgn(snr), PALETTE[0]),
                TheoryCurve("4-QAM theory", ser_qam4_awgn(snr), PALETTE[1])]

    # 3. SNR needed by each scheme at fixed SER: simulation and theory
    needed = {t: (snr_at_ser(SNR_DB, ser_bpsk, nerr_bpsk, t), snr_at_ser(SNR_DB, ser_qam4, nerr_qam4, t),
                  theory_snr_at_ser(ser_bpsk_awgn, t), theory_snr_at_ser(ser_qam4_awgn, t))
              for t in GAP_TARGETS}
    arrows = [GapArrow(needed[t][0], needed[t][1], t, f"{needed[t][1] - needed[t][0]:.1f} dB")
              for t in ARROW_TARGETS]                                  # measured BPSK -> 4-QAM gap

    # 4. Plot and save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_results(OUT_DIR / "awgn_ser", SNR_DB, sims, theories,
                 title="AWGN channel, ML detection, N = $10^6$ symbols per SNR point",
                 arrows=arrows)

    # 5. Numbers quoted in the discussion
    lines = agreement_lines([("BPSK vs BPSK theory", ser_bpsk, nerr_bpsk, theories[0].ser),
                             ("4-QAM vs 4-QAM theory", ser_qam4, nerr_qam4, theories[1].ser)],
                            N_SYMBOLS)
    lines += ["", "SNR (dB) needed for a target SER: simulation (theory in brackets)",
              f"{'SER':>8}{'BPSK':>16}{'4-QAM':>16}{'gap 4-QAM - BPSK':>20}"]
    for t, (b, q, b_th, q_th) in needed.items():
        lines.append(f"{t:>8.0e}{f'{b:.2f} ({b_th:.2f})':>16}{f'{q:.2f} ({q_th:.2f})':>16}"
                     f"{f'{q - b:.2f} ({q_th - b_th:.2f})':>20}")
    lines.append(f"High-SNR limit of the gap: 10 log10(2) = {10 * np.log10(2):.2f} dB")
    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "awgn_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
