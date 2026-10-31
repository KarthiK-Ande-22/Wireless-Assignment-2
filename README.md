# EC5.407 Wireless Communications: Assignment 2

**Detection and Diversity in AWGN and Fading Channels**: Monte Carlo simulation
of symbol detection, receive diversity and transmit diversity. Every simulated
SER curve is checked against theory.

| Problem | Topic | What is simulated |
|---|---|---|
| 1 | Detection in AWGN and fading | SER of BPSK and 4-QAM in AWGN (0–12 dB) and in Rayleigh fading (0–40 dB) |
| 2 | Receive diversity | 4-QAM over a 1 × L Rayleigh channel with MRC and selection combining (SC), L = 1, 2, 4 (0–24 dB) |
| 3 | Transmit diversity | 4-QAM over a 2 × 1 Rayleigh channel: single antenna, transmit beamforming and Alamouti, compared with MRC L = 2 (0–24 dB) |

The submitted report is [report/report.pdf](report/report.pdf). The assignment
statement is [WC_Assignment_2.pdf](WC_Assignment_2.pdf).

## Language, version and packages

- **Language:** Python 3.12 (tested with 3.12.3)
- **Packages:** numpy 2.3.4, scipy 1.16.3, matplotlib 3.10.7
  - scipy is used only for `scipy.special.erfc` (the Q-function), as the
    assignment allows.
  - The figures use the Computer Modern font that ships with matplotlib; no
    extra fonts are needed.
- **Report only:** a TeX distribution with `pdflatex` and the packages
  `pgfplots`/`pgfplotstable`, `booktabs`, `caption`, `placeins`, `enumitem` and
  `hyperref` (tested with TeX Live 2023).

```bash
pip install numpy scipy matplotlib
```

## Reproducing the figures

**Random seed: 2026.** Each script sets it once, at the start, with
`np.random.default_rng(2026)`; every random draw in that script uses this
generator.

Run every command from the project root. The scripts are run as modules
(`python3 -m ...`) so that they can import the shared `common` package.

| Command | Reproduces | Figures in the report | Runtime |
|---|---|---|---|
| `python3 -m problem1.awgn_ser` | Problem 1, Task 1 (AWGN) | Fig. 1 | ~6 s |
| `python3 -m problem1.rayleigh_ser` | Problem 1, Task 2 (Rayleigh fading) | Figs. 2 and 3 | ~13 s |
| `python3 -m problem2.receive_diversity` | Problem 2 (MRC and SC) | Figs. 4 and 5 | ~31 s |
| `python3 -m problem3.transmit_diversity` | Problem 3 (transmit diversity) | Figs. 6 and 7 | ~23 s |
| `python3 -m overall_check` | check of all points against exact theory (run last: it reads the CSV files of the other scripts) | Tables 1 and 2 | ~1 s |

Then build the report (it reads its figures and appendix tables from `results/`):

```bash
cd report
pdflatex report.tex
pdflatex report.tex    # second run resolves the cross-references
```

Each script prints its SER table and summary to the console and writes its
figures, data and summary to `results/`. Every figure and number in the report
comes from these scripts. Re-running them reproduces every output byte for byte.

## Repository layout

```
README.md                this file
WC_Assignment_2.pdf      assignment statement
overall_check.py         compares every plotted point of Problems 1-3 with exact theory

common/                  shared building blocks, used by every problem
  constellation.py         unit-energy BPSK and 4-QAM constellations
  channel.py               complex Gaussian samples; AWGN, Rayleigh, 1 x L SIMO and 2 x 1 MISO channels
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

results/                 all outputs of the scripts (see "Outputs")
  q1/  q2/  q3/            one folder per problem: figures (PDF), CSV tables, summaries
    images/                  PNG copy of every figure of that problem, for quick viewing
  overall_check.txt        output of overall_check.py

report/
  report.tex               LaTeX source of the report
  report.pdf               compiled report (20 pages)
```

## Outputs

| Folder | Files | Contents |
|---|---|---|
| `results/q1/` | `awgn_ser.{pdf,csv}`, `awgn_summary.txt` | AWGN SER figure (Task 1), SER table, agreement with theory, BPSK / 4-QAM gap |
| | `rayleigh_ser.{pdf,csv}`, `rayleigh_summary.txt` | Rayleigh SER figure (Task 2), SER table, agreement with theory, slopes, BPSK / 4-QAM gap, price of fading, share of errors in deep fades |
| | `awgn_vs_rayleigh.pdf` | extra figure: AWGN against fading, with the deep-fade probability |
| `results/q2/` | `receive_diversity.{pdf,csv}` | SER figure (Task 2) and table for MRC and SC, L = 1, 2, 4 |
| | `receive_diversity_theory.{pdf,csv}` | extra figure and table: the same points against exact theory |
| | `summary.txt` | L = 1 consistency check, slopes, array gains, SC − MRC gaps, agreement with exact theory, noise-variance check |
| `results/q3/` | `transmit_diversity.{pdf,csv}` | SER figure (Task 2) and table for all transmit schemes and MRC L = 2 |
| | `transmit_diversity_theory.{pdf,csv}` | extra figure and table: the same points against exact theory |
| | `summary.txt` | energy check, consistency checks, agreement with exact theory, slopes, array gains, SNR gaps |
| `results/q1–q3/images/` | one `.png` per figure | PNG copies of the figures above (same names), e.g. `results/q1/images/awgn_ser.png` |
| `results/` | `overall_check.txt` | every plotted point of all problems against exact theory: max \|z\| per curve and the statistics of z over all 160 independent points |

- **Figures:** each figure is saved twice by the same code. The vector `.pdf`
  (sharp at any zoom) is included in the report; the `.png` in `images/` is for
  quick viewing. Every figure has labelled axes and a legend; arrows mark the
  SNR gaps that the discussion explains. Only points with at least 10 errors are
  plotted.
- **CSV tables:** one row per SNR point. Each simulated curve has two
  columns, the SER (`sim`) and the error count (`n_err`). Each theory curve has
  one column.

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

so the same ML detector, `argmin_m |r - g u_m|^2`, is reused unchanged in all
three problems:

| Receiver | Output r | Gain g |
|---|---|---|
| no combining (Problem 1) | y | h |
| MRC (Problem 2) | hᴴy / ‖h‖ | ‖h‖ |
| SC (Problem 2) | y_k, with k = argmax \|h_k\|² | h_k |
| beamforming (Problem 3) | y | ‖h‖ |
| Alamouti (Problem 3) | (h₁* y₁ + h₂ y₂*) / ‖h‖, and the same for x₂ | ‖h‖ / √2 |

All schemes transmit a total energy of 1 per channel use, so the noise
variance 1/SNR means the same thing everywhere and the comparison is fair.

## Assignment coding requirements

| Requirement | How it is met |
|---|---|
| No built-in modulation, detection or error-rate functions | Constellations, ML detector, error counting and all SER formulas are written by hand in `common/` |
| Separate functions for each block | `bpsk`/`qam4` (constellation), `awgn_channel`/`rayleigh_channel`/`simo_rayleigh_channel`/`miso_channel` (channel), `ml_detect` (detection), `count_symbol_errors` (error counting), `qfunc` and the `ser_*` functions (theory) |
| Reuse, not copies | Every problem imports the same constellation and detector from `common/` |
| Vectorised code | All N symbols at an SNR point are processed at once with numpy arrays; the only loops are over SNR points, schemes and L |
| Seed set once per script | `rng = np.random.default_rng(2026)` at the start of `main()`; all random draws use this `rng` |
| N = 10⁶ symbols per SNR point | `N_SYMBOLS = 10**6` in every script (Alamouti uses N/2 blocks of two symbols) |
| Plot only points with at least 10 errors | `MIN_ERRORS = 10` in `common/plotting.py` |

## Main results

**All problems.** Every simulated curve agrees with its exact theory. Over all
160 independent plotted points, 71% lie within 1σ and 95% within 2σ (68% and
95% expected), with no bias. The largest deviation, 3.0σ, happens by chance in
one run in three.

**Problem 1.**
- The 4-QAM Rayleigh formula of the assignment table overestimates the SER by
  about 10%, because it leaves out the event that both I and Q are wrong. The
  exact expression, derived in the report, matches the simulation to within 1.8σ.
- In AWGN the SER falls exponentially; in fading only as 1/SNR (measured slope
  1.02), because errors come from deep fades: about three quarters of all fading
  errors occur during one. BPSK needs 17.3 dB more SNR than in AWGN at SER 10⁻³,
  and 25.8 dB more at 10⁻⁴.

**Problem 2.**
- MRC and SC both reach diversity order L.
- MRC has the larger array gain: 3.01 / 6.02 dB for L = 2 / 4, against
  1.76 / 3.19 dB for SC. SC therefore needs about 1.5 dB (L = 2) and
  3.2–3.5 dB (L = 4) more SNR than MRC at the same SER.

**Problem 3.**
- Beamforming, Alamouti and MRC L = 2 all have diversity order 2.
- Beamforming matches MRC L = 2 (gap within ±0.2 dB).
- Alamouti is 3.0 dB behind both, because without channel knowledge at the
  transmitter it has no array gain.
- The single antenna needs 6.3 dB more SNR than Alamouti at SER 10⁻², and the
  gap grows as the SER falls (diversity 1 against 2).

The full numbers are in each `summary.txt` and in the report.

## Report structure

| Section | Contents |
|---|---|
| 1 | Simulation set-up: conventions, code structure, accuracy of a simulated point, exact theory, overall check |
| 2 | Problem 1: Task 1 (AWGN), Task 2 (Rayleigh fading), discussion |
| 3 | Problem 2: Task 1 (MRC and SC as scalar channels), Task 2 (simulation), discussion |
| 4 | Problem 3: Task 1 (single antenna, beamforming, Alamouti, energy check), Task 2 (simulation), discussion |
| 5 | Summary of the SNR gaps marked in the figures |
| A | Appendix: all simulated SER values next to theory |

Every figure caption states the SNR range and the number of Monte Carlo symbols
per SNR point (N = 10⁶).
