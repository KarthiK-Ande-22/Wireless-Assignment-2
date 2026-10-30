# Problem 1: Symbol Error Rate in AWGN and Fading Channels

All numbers below come from `awgn_ser.csv` and `rayleigh_ser.csv` in this
folder (seed 2026, N = 10^6 symbols per SNR point).

## Figures

- **Fig. 1, `awgn_ser.png`:** SER of BPSK and 4-QAM with ML detection in the AWGN
  channel (h = 1), SNR = 0, 1, ..., 12 dB, N = 10^6 Monte Carlo symbols per SNR
  point. Markers show simulation and lines show theory. Points with fewer than
  10 errors are not plotted.
- **Fig. 2, `rayleigh_ser.png`:** SER of BPSK and 4-QAM with coherent ML detection
  in Rayleigh fading (h ~ CN(0, 1), drawn independently per symbol), SNR = 0, 2,
  ..., 40 dB, N = 10^6 Monte Carlo symbols per SNR point. The solid 4-QAM line is
  the approximation from the assignment table. The dotted line is the exact
  expression derived below.

## Implementation

| Block | Implementation |
|---|---|
| Constellation | BPSK {-1, +1}; 4-QAM {±1/√2 ± j/√2}; both have E[\|x\|²] = 1 (checked numerically) |
| Channel | y = hx + w, w ~ CN(0, 1/SNR); h = 1 (AWGN) or h ~ CN(0, 1), new for every symbol |
| Detector | ML = minimum distance to the scaled constellation, î = argmin_m \|y − h u_m\|², vectorised over an N × M distance array |
| Error counting | n_err = #{î ≠ i}, comparing indices; P̂e = n_err / N |
| Theory | Q(a) = ½ erfc(a/√2), with the expressions from the assignment table |

## Agreement between simulation and theory

The Monte Carlo estimate has relative standard deviation std(P̂e)/Pe ≈ 1/√n_err.
Deviations are therefore measured in units of that accuracy (σ).

**AWGN (Fig. 1).** Both curves lie on the theory to within Monte Carlo accuracy.
Where n_err > 10³, the deviation is a few percent. All plotted points are within
2.6σ, and the mean relative deviation is +3% for BPSK and −1% for 4-QAM.
Agreement is visibly poorer only at the highest plotted SNR, where few errors
are observed. For example, BPSK at 9 dB has 40 errors, so the accuracy is
about ±16%, and the observed +19% is about 1σ. BPSK at 10, 11 and 12 dB gave
6, 0 and 0 errors (Pe = 3.9·10⁻⁶ … 9·10⁻⁹). Those points are omitted under the
n_err ≥ 10 rule, because 10⁶ symbols cannot resolve them.

**Rayleigh fading, BPSK (Fig. 2).** Simulation and the exact expression
Pe = ½(1 − √(SNR/(1+SNR))) agree to within statistical accuracy. The mean
deviation is +1.4% and every point is within 3σ.

**Rayleigh fading, 4-QAM (Fig. 2): a systematic deviation.** The simulated SER
lies consistently below the table approximation: 13.5% lower at 0 dB, falling
to about 9% lower from roughly 15 dB upward. At 0 dB the accuracy is ±0.2%,
so this is about 80σ and not statistical noise. In other words, the
approximation overestimates the true SER by 16% at 0 dB and by about 10% at
high SNR. Above about 30 dB, the points scatter by ±5–10% around this offset
because only a few hundred errors are observed there.

The reason is that the approximation keeps only part of the exact error
probability. Conditioned on h, let q(h) = Q(√(SNR\|h\|²)) be the error
probability of one quadrature component. Then

  Pe = E[2q(h) − q(h)²] = 2 · P_BPSK(SNR/2) − E[q(h)²].

The approximation keeps only the first term and drops the "both I and Q in
error" term E[q²], so it is an upper bound.

To get the exact value, write Q² in Craig's form,
Q²(x) = (1/π)∫₀^{π/4} exp(−x²/(2 sin²θ)) dθ, and average over \|h\|² ~ Exp(1):

  E[q²] = ¼ [1 − (4/π) μ arctan(1/μ)],  μ = √((SNR/2)/(1 + SNR/2)),

  Pe(exact) = (1 − μ) − ¼ [1 − (4/π) μ arctan(1/μ)].

The simulation follows this exact curve (dotted) to within statistical
accuracy: the mean deviation is −1.1% and every point is within 1.8σ.

The dropped term is not small in fading, even at high SNR. As μ → 1,
E[q²] ≈ ¼(1 − 2/π)(1 − μ), which is about 9% of the approximation at every high
SNR. That is a constant offset of about 0.4 dB along the SNR axis. In AWGN, by
contrast, Q² ≪ Q at high SNR and the term vanishes. The difference comes from
when errors occur. In fading, errors occur almost only during deep fades, where
SNR·\|h\|² is small, and then I and Q tend to fail together.

## AWGN vs. fading at high SNR: the deep-fade argument

The two figures differ sharply in shape:

- **AWGN: "waterfall".** Q(x) ≤ ½ e^{−x²/2}, so Pe decays exponentially in SNR
  and the log-scale curve gets steeper and steeper. Around 11 dB the 4-QAM curve
  falls at about 7 decades per 10 dB.
- **Rayleigh: straight line.** For large SNR, Pe ≈ 1/(4·SNR) for BPSK and
  ≈ 1/SNR for 4-QAM, so the curve is a straight line of slope −1 decade per
  decade. The measured slope over 30–40 dB is −1.02 for both schemes, which
  means diversity order 1.

The practical cost is large. Reaching Pe = 10⁻⁴ with BPSK requires
Q(√(2·SNR)) = 10⁻⁴, that is SNR ≈ 8.4 dB, in AWGN. In Rayleigh fading it
requires 1/(4·SNR) = 10⁻⁴, that is SNR ≈ 34 dB: a penalty of about 26 dB. At
10 dB, BPSK has Pe = 3.9·10⁻⁶ in AWGN but 2.3·10⁻² in fading.

The deep-fade argument [1, Sec. 5.4] explains this. Conditioned on the channel,
Pe(h) = Q(√(2·SNR\|h\|²)):

- When SNR·\|h\|² ≫ 1 (a good channel), Pe(h) is exponentially small. Noise
  alone almost never causes an error.
- When SNR·\|h\|² ≲ 1 (a deep fade), Pe(h) is of order one.

The average error probability is therefore set by the probability of a deep
fade. Since \|h\|² ~ Exp(1),

  Pr[\|h\|² < 1/SNR] = 1 − e^{−1/SNR} ≈ 1/SNR,

so Pe ∝ SNR⁻¹. In fading, errors are caused by the channel being in a deep fade,
not by noise on a good channel. The probability of a deep fade falls only
linearly with SNR, so a tenfold increase in power buys only a tenfold
reduction in Pe. That is why the fading curves have slope −1 instead of
the waterfall. It also motivates diversity (Problems 2 and 3), which lowers
the probability that all copies of the signal fade at once.
