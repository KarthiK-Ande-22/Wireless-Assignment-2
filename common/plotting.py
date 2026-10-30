"""Shared output for SER experiments: figure (G.7), CSV table and console summary."""
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")   # render to files only, no display needed
import matplotlib.pyplot as plt
import numpy as np

MIN_ERRORS = 10   # plot only points backed by at least this many errors (G.7)

# Fixed categorical colour order: blue, orange, aqua, violet.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]

# Typeset figure text in Computer Modern, the font of the LaTeX report.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],                # Computer Modern, shipped with matplotlib
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,    # tick labels 10^{-k} in the same font
    "axes.unicode_minus": False,            # cmr10 has no Unicode minus sign
    "font.size": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})


@dataclass
class SimCurve:
    """Monte Carlo SER of one scheme, drawn as markers (optionally joined by a line)."""
    name: str                 # CSV column prefix
    ser: np.ndarray
    n_errors: np.ndarray
    color: str
    marker: str
    linestyle: str = "none"   # "none": markers only, theory is drawn separately
    label: str = None         # legend text, defaults to "<name> simulation"


@dataclass
class TheoryCurve:
    """Theoretical SER, drawn as a line."""
    label: str
    ser: np.ndarray
    color: str
    linestyle: str = "-"
    linewidth: float = 1.6
    in_legend: bool = True    # False: drawn but left out of the legend
    snr_db: np.ndarray = None # own SNR grid (plot_ser only); default: the figure's grid


@dataclass
class GapArrow:
    """Double-headed arrow marking the horizontal SNR gap between two curves at one SER."""
    snr_from: float           # dB, left end (better curve)
    snr_to: float             # dB, right end (worse curve)
    ser: float                # SER level at which the gap is read
    label: str                # e.g. "3.9 dB"
    label_side: str = "above" # where the label goes: "above", "below", "left" or "right" of the arrow


INK = "#333333"   # arrows and annotation text: neutral ink, not a series colour


def plot_ser(path_stem, snr_db, sims, theories, title, ylim=(1e-6, 1),
             figsize=(7, 5), legend_loc="lower left", arrows=(), xlabel="SNR (dB)"):
    """Semilogy plot of theory (lines) and simulation (open markers); saved as PNG and PDF.

    `arrows` mark the horizontal SNR gaps between curves. The legend sits inside
    the axes at `legend_loc`.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for c in theories:
        ax.semilogy(snr_db if c.snr_db is None else c.snr_db, c.ser, color=c.color, linestyle=c.linestyle,
                    linewidth=c.linewidth, label=c.label if c.in_legend else "_nolegend_")
    for c in sims:
        reliable = c.n_errors >= MIN_ERRORS
        ax.semilogy(snr_db[reliable], c.ser[reliable], linestyle=c.linestyle,
                    linewidth=1.5, marker=c.marker, markersize=7, color=c.color,
                    markerfacecolor="none", markeredgewidth=1.5,          # open markers: simulation
                    label=c.label or f"{c.name} simulation")
    for a in arrows:
        ax.annotate("", xy=(a.snr_to, a.ser), xytext=(a.snr_from, a.ser),
                    arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.2,
                                    shrinkA=0, shrinkB=0))
        x, y, ha, va = {"above": ((a.snr_from + a.snr_to) / 2, a.ser * 1.25, "center", "bottom"),
                        "below": ((a.snr_from + a.snr_to) / 2, a.ser / 1.25, "center", "top"),
                        "left": (a.snr_from - 0.025 * (snr_db[-1] - snr_db[0]), a.ser, "right", "center"),
                        "right": (a.snr_to + 0.055 * (snr_db[-1] - snr_db[0]), a.ser, "left", "center"),
                        }[a.label_side]
        ax.text(x, y, a.label, color=INK, fontsize=10, ha=ha, va=va)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Symbol error rate $\hat{P}_e$")
    ax.set_title(title)
    pad = 0.02 * (snr_db[-1] - snr_db[0])   # keep edge markers fully visible
    ax.set_xlim(snr_db[0] - pad, snr_db[-1] + pad)
    ax.set_ylim(*ylim)
    ax.grid(True, which="major", color="#c8c8c8", linewidth=0.7)
    ax.grid(True, which="minor", color="#e8e8e8", linewidth=0.4)
    ax.legend(loc=legend_loc, framealpha=1, edgecolor="#999999")
    fig.tight_layout()
    fig.savefig(f"{path_stem}.png", dpi=200)
    fig.savefig(f"{path_stem}.pdf", metadata={"CreationDate": None})   # no timestamp: reproducible bytes
    plt.close(fig)


def save_results(path_stem, snr_db, sims, theories, title, **plot_kwargs):
    """Write `<path_stem>.csv`, `.png` and `.pdf`, and print the SER table."""
    columns = {"snr_db": snr_db}
    for c in sims:
        columns[f"{c.name} sim"] = c.ser
        columns[f"{c.name} n_err"] = c.n_errors
    for c in theories:
        columns[c.label] = c.ser

    np.savetxt(f"{path_stem}.csv", np.column_stack(list(columns.values())),
               delimiter=",", header=",".join(columns), comments="", fmt="%.6g")
    plot_ser(path_stem, snr_db, sims, theories, title, **plot_kwargs)

    print(title.replace("$", ""))
    widths = [max(10, len(name) + 2) for name in columns]   # fit each column to its header
    print("".join(f"{name:>{w}}" for name, w in zip(columns, widths)))
    for row in zip(*columns.values()):
        print("".join(f"{value:>{w}.4g}" for value, w in zip(row, widths)))
    print(f"Saved {path_stem}.{{csv,png,pdf}}\n")
