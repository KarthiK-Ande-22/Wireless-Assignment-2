"""Figures, CSV tables and printed tables for the SER results."""
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # save figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

MIN_ERRORS = 10   # plot only points with at least 10 errors (assignment rule)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]   # blue, orange, green, violet

# Figure text in Computer Modern, the font of the report
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,   # cmr10 has no Unicode minus sign
    "font.size": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
})


@dataclass
class SimCurve:
    """Simulated SER of one scheme, drawn with markers."""
    name: str                 # also used as the CSV column name
    ser: np.ndarray
    n_errors: np.ndarray
    color: str
    marker: str
    linestyle: str = "none"
    label: str = None         # legend text


@dataclass
class TheoryCurve:
    """Theoretical SER, drawn as a line."""
    label: str
    ser: np.ndarray
    color: str
    linestyle: str = "-"
    linewidth: float = 1.6
    in_legend: bool = True
    snr_db: np.ndarray = None # own SNR grid, if different from the figure's


@dataclass
class GapArrow:
    """Arrow showing the SNR gap between two curves at one SER level."""
    snr_from: float
    snr_to: float
    ser: float
    label: str
    label_side: str = "above" # "above", "below", "left" or "right"


INK = "#333333"   # colour of the arrows and their labels


def plot_ser(path_stem, snr_db, sims, theories, title, ylim=(1e-6, 1),
             figsize=(7, 5), legend_loc="lower left", arrows=(), xlabel="SNR (dB)"):
    """Plot SER curves: theory as lines, simulation as markers.

    Saves `<path_stem>.pdf` for the report and a PNG copy in the `images/` folder next to it.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for c in theories:
        ax.semilogy(snr_db if c.snr_db is None else c.snr_db, c.ser, color=c.color, linestyle=c.linestyle,
                    linewidth=c.linewidth, label=c.label if c.in_legend else "_nolegend_")
    for c in sims:
        reliable = c.n_errors >= MIN_ERRORS
        ax.semilogy(snr_db[reliable], c.ser[reliable], linestyle=c.linestyle,
                    linewidth=1.5, marker=c.marker, markersize=7, color=c.color,
                    markerfacecolor="none", markeredgewidth=1.5,
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
    pad = 0.02 * (snr_db[-1] - snr_db[0])   # keep the edge markers fully visible
    ax.set_xlim(snr_db[0] - pad, snr_db[-1] + pad)
    ax.set_ylim(*ylim)
    ax.grid(True, which="major", color="#c8c8c8", linewidth=0.7)
    ax.grid(True, which="minor", color="#e8e8e8", linewidth=0.4)
    ax.legend(loc=legend_loc, framealpha=1, edgecolor="#999999")
    fig.tight_layout()
    fig.savefig(f"{path_stem}.pdf", metadata={"CreationDate": None})   # no date, so re-runs give identical files
    image_dir = Path(path_stem).parent / "images"
    image_dir.mkdir(exist_ok=True)
    fig.savefig(image_dir / f"{Path(path_stem).name}.png", dpi=200)
    plt.close(fig)


def save_results(path_stem, snr_db, sims, theories, title, **plot_kwargs):
    """Save the CSV table and the figure, and print the table."""
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
    widths = [max(10, len(name) + 2) for name in columns]
    print("".join(f"{name:>{w}}" for name, w in zip(columns, widths)))
    for row in zip(*columns.values()):
        print("".join(f"{value:>{w}.4g}" for value, w in zip(row, widths)))
    print(f"Saved {path_stem}.{{csv,pdf}} and images/{Path(path_stem).name}.png\n")
