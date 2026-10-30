# Problem 2: Receive Diversity

All numbers below come from `receive_diversity.csv` and `summary.txt` in this
folder. Both are produced by `python3 -m problem2.receive_diversity` with seed
2026 and N = 10^6 symbols per SNR point.

## Figure

**Fig. 3, `receive_diversity.png`:** SER of 4-QAM over the 1 × L SIMO Rayleigh
channel with maximal ratio combining (MRC, solid, circles) and selection
combining (SC, dashed, squares), for L = 1, 2, 4. Each value of L has one colour.
SNR = 0, 2, ..., 24 dB (average SNR per receive antenna), with N = 10^6 Monte
Carlo symbols per SNR point. Points with fewer than 10 errors are omitted. The
grey band is the exact 4-QAM Rayleigh SER of Problem 1. For each L, MRC and SC
process the same channel and noise realisations.

## Implementation

| Block | Code | Implementation |
|---|---|---|
| Constellation, detector | `common/constellation.py`, `common/detector.py` | reused unchanged from Problem 1 |
| SIMO channel | `common/channel.py: simo_rayleigh_channel` | y_l = h_l x + w_l, with h_l ~ CN(0, 1) i.i.d. and w_l ~ CN(0, 1/SNR), new for every symbol; arrays of shape (N, L) |
| MRC | `common/combining.py: mrc_combine` | r = h^H y / ‖h‖, effective gain ‖h‖ |
| SC | `common/combining.py: selection_combine` | l* = argmax_l \|h_l\|², r = y_l*, effective gain h_l* |
| Monte Carlo | `common/simulation.py: simulate_receivers` | both receivers detect the same realisations |

## Decision statistics as equivalent scalar channels

Both combiners turn the L-antenna observation into a scalar channel

  r = g x + n,  n ~ CN(0, 1/SNR),

which has the same form as the fading channel of Problem 1. The ML detector of
Problem 1, î = argmin_m \|r − g u_m\|², is therefore applied unchanged.

**MRC.** The combined statistic is

  r = h^H y / ‖h‖ = ‖h‖ x + h^H w / ‖h‖.

- Effective gain: g = ‖h‖ = (Σ_l \|h_l\|²)^{1/2}, which is real and positive.
- Noise variance: Var(h^H w / ‖h‖) = Σ_l \|h_l\|² (1/SNR) / ‖h‖² = 1/SNR.
- Instantaneous post-combining SNR: γ_MRC = ‖h‖² · SNR = Σ_l \|h_l\|² · SNR.

Our implementation normalises by ‖h‖. The unnormalised statistic
r′ = h^H y = ‖h‖² x + h^H w has gain ‖h‖² and noise variance ‖h‖²/SNR, so its
SNR is the same. Detecting r′ with gain ‖h‖² is also equivalent to ML:

- The metric is a positive per-symbol rescaling of the normalised one,
  \|r′ − ‖h‖² u\|² = ‖h‖² · \|r − ‖h‖ u\|², so the argmin is the same.
- The normalised metric equals the full vector ML metric up to a term that does
  not depend on u: ‖y − h u‖² = \|r − ‖h‖ u\|² + (‖y‖² − \|r\|²).

So MRC followed by the scalar minimum-distance rule *is* the ML detector for
the vector observation y. The MRC output is a sufficient statistic.

**SC.** The receiver selects l* = argmax_l \|h_l\|² and uses

  r = y_l* = h_l* x + w_l*.

- Effective gain: g = h_l*, which is complex. Coherent detection uses its phase.
- Noise variance: 1/SNR. The selection depends only on h, never on w, so w_l*
  is still CN(0, 1/SNR) and independent of x.
- Instantaneous post-combining SNR: γ_SC = max_l \|h_l\|² · SNR.

## Consistency check (L = 1)

For L = 1, MRC reduces to a phase rotation r = (h*/\|h\|) y, and SC has only
one branch to choose. Both metrics equal \|y − h u\|², the coherent detector of
Problem 1. The decisions are therefore identical, not just statistically
equivalent:

- The MRC and SC error counts for L = 1 are identical at every SNR (`summary.txt`).
- An independent re-run of the Problem 1 fading simulation agrees with the
  L = 1 curve to within 1.7σ.
- The exact 4-QAM Rayleigh expression of Problem 1 agrees to within 1.9σ.

In the figure, the L = 1 curves lie on the grey Problem 1 band. This confirms
that the multi-antenna channel and the combiners are implemented consistently
with Problem 1.

## Slope at high SNR and diversity order

[2] predicts diversity order L for both MRC and SC, that is SER ∝ SNR^{−L}.
This follows from the deep-fade argument of Problem 1, extended to L branches.
An error is likely only when the post-combining SNR falls below about 1, and
that requires all L independent branches to fade at the same time:

  Pr[‖h‖² < 1/SNR] ≈ SNR^{−L} / L!   (MRC),
  Pr[max_l \|h_l\|² < 1/SNR] ≈ SNR^{−L}   (SC).

The local slopes measured over the last 2 dB steps, −Δlog₁₀ SER / Δlog₁₀ SNR,
are listed in `summary.txt`:

| L | MRC | SC | predicted |
|---|---|---|---|
| 1 | 1.02 → 0.95 → 0.94 (18–24 dB) | identical to MRC | 1 |
| 2 | 1.97 → 2.07 → 1.99 (16–22 dB) | 1.98 → 1.78 → 2.42 (16–22 dB) | 2 |
| 4 | 2.60 → 2.78 → 3.40 (4–10 dB) | 2.59 → 3.15 → 3.22 (8–14 dB) | 4 |

- **L = 1 and L = 2:** the measured slopes match the diversity order. With a few
  hundred errors per point, a single 2 dB step has an uncertainty of about
  ±0.1–0.3. This explains the scatter in the SC L = 2 row.
- **L = 4:** the slope is still increasing, from about 2.6 to about 3.3. The
  asymptotic regime SER ∝ SNR^{−4} sets in only at SER levels below what
  10^6 symbols can measure. The L = 4 curves reach Pe ≈ 10^{−5} at 14–16 dB, and
  beyond that they produce fewer than 10 errors. The steepening trend and the
  visibly faster fall-off with L are consistent with diversity order 4.

## SNR gap between MRC and SC, and array gain

For each L, MRC and SC have the same diversity order. Their curves are
therefore parallel at high SNR and separated by a roughly constant horizontal
SNR gap. Read off the simulated curves (`summary.txt`):

| L | gap at SER 10⁻² | 10⁻³ | 10⁻⁴ | array-gain difference | high-SNR limit |
|---|---|---|---|---|---|
| 2 | 1.50 dB | 1.46 dB | 1.32 dB | 1.25 dB | 1.51 dB |
| 4 | 3.26 dB | 3.20 dB | 3.47 dB | 2.83 dB | 3.45 dB |

**Origin: array gain [2, Sec. 3.4].** The array gain is the increase in average
post-combining SNR over a single antenna.

- **MRC** adds the signal energy of all L branches coherently:
  E[γ_MRC]/SNR = E‖h‖² = L. That is 3.01 dB for L = 2 and 6.02 dB for L = 4.
- **SC** keeps the energy of only the strongest branch and discards the other
  L − 1: E[γ_SC]/SNR = E[max_l \|h_l\|²] = Σ_{k=1}^{L} 1/k. That is 1.76 dB for
  L = 2 and 3.19 dB for L = 4.

The Monte Carlo estimates in `summary.txt` reproduce all four values to 0.01 dB.
The difference, 1.25 dB for L = 2 and 2.83 dB for L = 4, accounts for most of
the measured gap. It also explains why the gap widens with L: MRC's array gain
grows linearly in L, while SC's grows only like the harmonic number,
H_L ≈ ln L + 0.58.

The measured gap is slightly larger than the array-gain difference. This is
because the SER at high SNR depends on the lower tail of the post-combining SNR
distribution (the deep-fade probability), not only on its mean. From the two
deep-fade probabilities above, SC fades below a given level L! times more often
than MRC. SER_SC/SER_MRC therefore tends to L!, which on curves of slope L
corresponds to a gap of 10 log₁₀(L!)/L: 1.51 dB for L = 2 and 3.45 dB for L = 4.
The measured gaps lie between the array-gain difference and this limit, within
the accuracy of reading the gap off curves sampled every 2 dB (about ±0.1–0.2 dB).

In short:

- Both schemes achieve full diversity L.
- MRC also collects the full array gain L, while SC collects only H_L.
- That missing array gain is the price SC pays for needing only a single RF
  chain and an antenna switch.
