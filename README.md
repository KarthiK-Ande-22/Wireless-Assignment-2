# EC5.407 Wireless Communications: Assignment 2

Monte Carlo simulation of symbol detection and diversity in AWGN and Rayleigh
fading channels. Every simulated SER curve is checked against theory.

| Problem | Topic | What is simulated |
|---|---|---|
| 1 | Detection in AWGN and fading | SER of BPSK and 4-QAM in AWGN (0–12 dB) and in Rayleigh fading (0–40 dB) |
| 2 | Receive diversity | 4-QAM over a 1 × L Rayleigh channel with MRC and selection combining (SC), L = 1, 2, 4 (0–24 dB) |
| 3 | Transmit diversity | 4-QAM over a 2 × 1 Rayleigh channel: single antenna, transmit beamforming and Alamouti, compared with MRC L = 2 (0–24 dB) |

The submitted report is [report/report.pdf](report/report.pdf). The assignment
statement (`WC_Assignment_2.pdf`, EC5.407 course material) is not included in
this repository.

## Requirements

- Python 3.12 (tested with 3.12.3)
- numpy 2.3.4
- scipy 1.16.3 (only `scipy.special.erfc` is used, for the Q-function)
- matplotlib 3.10.7
- For the report only: a TeX distribution with `pdflatex` and the `pgfplots`
  package (tested with TeX Live 2023)

Install the Python packages with:

```bash
pip install numpy scipy matplotlib
```

## Quick start

Run every command from the project root. The scripts are run as modules
(`python3 -m ...`) so that they can import the shared `common` package.

```bash
python3 -m problem1.awgn_ser             # Problem 1, Task 1 (AWGN)       ~6 s
python3 -m problem1.rayleigh_ser         # Problem 1, Task 2 (Rayleigh)   ~13 s
python3 -m problem2.receive_diversity    # Problem 2 (MRC and SC)         ~31 s
python3 -m problem3.transmit_diversity   # Problem 3 (transmit diversity) ~23 s
python3 -m overall_check                 # all points vs exact theory     ~1 s (run last)

cd report && pdflatex report.tex && pdflatex report.tex   # build the report
```

Each script prints its SER table and summary to the console. It also writes
its figures, data and summary to `results/`. Every figure and number in the
report comes from these scripts.

## Repository layout

```
README.md                this file
overall_check.py         compares every plotted point of Problems 1-3 with exact theory

common/                  shared building blocks, used by every problem
  constellation.py         unit-energy BPSK and 4-QAM constellations
  channel.py               complex Gaussian samples; AWGN, Rayleigh, 1 x L SIMO and MISO channels
  detector.py              vectorised ML (minimum-distance) detector
  combining.py             receive combiners (none, MRC, SC)
  theory.py                Q-function, the SER expressions of the assignment table and
                           the exact 4-QAM SER of every scheme (single antenna, MRC, SC,
                           beamforming, Alamouti)
  simulation.py            dB conversion, error counting, Monte Carlo loop (Algorithm 1)
  plotting.py              SER figures (with gap arrows), CSV tables and console tables
  analysis.py              SNR at a given SER, SNR gaps, high-SNR slopes, deviations in sigma

problem1/
  awgn_ser.py              Task 1: BPSK and 4-QAM in AWGN
  rayleigh_ser.py          Task 2: BPSK and 4-QAM in Rayleigh fading
problem2/
  receive_diversity.py     MRC and SC with L = 1, 2, 4
problem3/
  transmit_diversity.py    single antenna, beamforming, Alamouti; MRC L = 2 for comparison

results/
  q1/                      Problem 1 outputs
  q2/                      Problem 2 outputs
  q3/                      Problem 3 outputs

report/
  report.tex               LaTeX source of the report
  report.pdf               compiled report (16 pages)
```

## Outputs

| Folder | Files | Contents |
|---|---|---|
| `results/q1/` | `awgn_ser.{png,pdf,csv}`, `awgn_summary.txt` | AWGN SER figure (Task 1), SER table, agreement with theory, BPSK / 4-QAM gap |
| | `rayleigh_ser.{png,pdf,csv}`, `rayleigh_summary.txt` | Rayleigh SER figure (Task 2), SER table, agreement with theory, slopes, BPSK / 4-QAM gap, price of fading, share of errors in deep fades |
| | `awgn_vs_rayleigh.{png,pdf}` | extra figure: AWGN against fading, with the deep-fade probability |
| | `discussion.md` | answers to the Problem 1 questions |
| `results/q2/` | `receive_diversity.{png,pdf,csv}` | SER figure (Task 2) and table for MRC and SC, L = 1, 2, 4 |
| | `receive_diversity_theory.{png,pdf,csv}` | extra figure and table: the same points against exact theory |
| | `summary.txt` | L = 1 consistency check, slopes, array gains, SC − MRC gaps, agreement with exact theory, noise-variance check |
| | `discussion.md` | answers to the Problem 2 questions |
| `results/q3/` | `transmit_diversity.{png,pdf,csv}` | SER figure (Task 2) and table for all transmit schemes and MRC L = 2 |
| | `transmit_diversity_theory.{png,pdf,csv}` | extra figure and table: the same points against exact theory |
| | `summary.txt` | energy check, consistency checks, agreement with exact theory, slopes, array gains, SNR gaps |
| | `discussion.md` | answers to the Problem 3 questions |
| `results/` | `overall_check.txt` | every plotted point of all problems against exact theory: max \|z\| per curve and the statistics of z over all 160 independent points |

- **Figures:** `.png` for viewing, `.pdf` for the report. Arrows on the main
  figures mark the SNR gaps between curves that the discussion explains.
- **CSV tables:** one row per SNR point. Each simulated curve has two
  columns, the SER (`sim`) and the error count (`n_err`). Each plotted theory
  curve has one column.

## How the code is organised

Each simulation follows the same chain:

```
constellation -> (transmitter) -> channel -> receiver / combiner -> ML detector -> error count
```

Every receiver in the assignment reduces the received signal to one
**equivalent scalar channel** per symbol,

```
r = g x + n,   n ~ CN(0, 1/SNR)
```

The same ML detector, `argmin_m |r - g u_m|^2`, is therefore reused unchanged
in all three problems:

| Receiver | Output r | Gain g |
|---|---|---|
| no combining (Problem 1) | y | h |
| MRC (Problem 2) | hᴴy / ‖h‖ | ‖h‖ |
| SC (Problem 2) | y_k, with k = argmax \|h_k\|² | h_k |
| beamforming (Problem 3) | y | ‖h‖ |
| Alamouti (Problem 3) | (h₁* y₁ + h₂ y₂*) / ‖h‖, and the same for x₂ | ‖h‖ / √2 |

All schemes transmit a total energy of 1 per channel use, so the noise
variance 1/SNR means the same thing everywhere and the comparison is fair.

## Assignment requirements

| Requirement | How it is met |
|---|---|
| No built-in modulation, detection or error-rate functions | Constellations, ML detector, error counting and all SER formulas are written by hand in `common/` |
| Vectorised code | All N symbols at an SNR point are processed at once with numpy arrays; the only loops are over SNR points and schemes |
| Seed set once per script | `rng = np.random.default_rng(2026)` at the start of `main()`; all random draws use this `rng` |
| N = 10⁶ symbols per SNR point | `N_SYMBOLS = 10**6` in every script (Alamouti uses N/2 blocks of two symbols) |
| Plot only points with at least 10 errors | `MIN_ERRORS = 10` in `common/plotting.py`; points with fewer errors are left out of the figures |
| Separate functions for each block | Constellation, channel, detection, error counting and theory each have their own module; `qfunc` in `common/theory.py` is a separate function |

## Main results

**All problems.** Every simulated curve agrees with its exact theory. Over all
160 independent plotted points, 71% lie within 1σ and 95% within 2σ (68% and
95% expected), with no bias. The largest deviation, 3.0σ, happens by chance in
one run in three. MRC and SC with L = 1 and the Problem 3 single-antenna curve
use the same random numbers, so they are counted once.

**Problem 1.**
- All simulated curves agree with theory to within about 3σ of the Monte Carlo
  accuracy.
- The 4-QAM Rayleigh approximation in the assignment table overestimates the
  SER by about 10%. The exact expression, derived in the report, matches the
  simulation to within 1.8σ.
- In fading, both modulations have diversity order 1 (measured slope 1.02).
  BPSK needs 17.3 dB more SNR than in AWGN at SER 10⁻³, and 25.8 dB more at
  10⁻⁴. About three quarters of all fading errors occur during a deep fade.

**Problem 2.**
- MRC and SC both reach diversity order L.
- MRC has the larger array gain: 3.01 / 6.02 dB for L = 2 / 4, against
  1.76 / 3.19 dB for SC.
- SC needs about 1.5 dB (L = 2) and 3.3 dB (L = 4) more SNR than MRC at the
  same SER.

**Problem 3.**
- Beamforming, Alamouti and MRC L = 2 all have diversity order 2.
- Beamforming matches MRC L = 2 (gap within ±0.2 dB).
- Alamouti is 3.0 dB worse than both, because it has no array gain without
  channel knowledge at the transmitter.
- The single antenna needs 6.3 dB more SNR than Alamouti at SER 10⁻².

The full numbers are in each `summary.txt`. The explanations are in each
`discussion.md` and in the report.

## Report

`report/report.tex` reads its figures and the SER tables of its appendix
directly from `results/`. Run the scripts first, then build it:

```bash
cd report
pdflatex report.tex
pdflatex report.tex    # second run resolves the cross-references
```

## Reproducibility

- Random seed **2026** in every script.
- Re-running a script reproduces its CSV, summary, PNG and PDF outputs
  byte for byte. The figure PDFs are saved without a creation date for this
  reason.
- Runtimes measured on a laptop: about 6 s (AWGN) and 13 s (Rayleigh) for
  Problem 1, 31 s for Problem 2 and 23 s for Problem 3.
