# Problem 3: Transmit Diversity

All numbers below come from `transmit_diversity.csv` and `summary.txt` in this
folder. Both are produced by `python3 -m problem3.transmit_diversity` with seed
2026 and N = 10^6 symbols per SNR point.

## Figure

**Fig. 4, `transmit_diversity.png`:** SER of 4-QAM over the 2 × 1 MISO Rayleigh
channel (h₁, h₂ ~ CN(0, 1) i.i.d., known at the receiver) for three schemes:

- single transmit antenna (reference),
- transmit beamforming (channel known at the transmitter),
- the Alamouti code (no channel knowledge at the transmitter).

MRC with L = 2 receive antennas (Problem 2, dashed) is shown for comparison.
SNR = 0, 2, ..., 24 dB, with noise variance 1/SNR referred to a total
transmitted energy of one per channel use. N = 10^6 Monte Carlo symbols per SNR
point; Alamouti uses N/2 blocks of two channel uses. Points with fewer than 10
errors are omitted.

## Implementation

| Block | Code | Implementation |
|---|---|---|
| Constellation, detector | `common/constellation.py`, `common/detector.py` | reused unchanged from Problem 1 |
| MISO channel | `common/channel.py: miso_channel` | y = Σ_l h_l s_l + w, w ~ CN(0, 1/SNR); h drawn once per block, constant over the block's channel uses, independent across blocks |
| Transmitters | `problem3/transmit_diversity.py: single_antenna_tx, beamforming_tx, alamouti_tx` | map symbols (and h, for beamforming only) to the signal each antenna sends in each channel use |
| Receivers | `no_combining` (from `common/combining.py`), `beamforming_rx`, `alamouti_rx` | reduce y to one equivalent scalar channel r = g x + n per symbol |
| Monte Carlo | `simulate_scheme` | Algorithm 1 with a transmitter stage; also measures the transmitted energy and E\|g\|² |
| MRC, L = 2 | `simulate_receivers` + `mrc_combine` | the same code as Problem 2 |

Every receiver output has the form r = g x + n with n ~ CN(0, 1/SNR). The ML
detector of Problem 1, î = argmin_m \|r − g u_m\|², is therefore applied
unchanged to all schemes.

## Transmission schemes, scaling and decision statistics

**Single antenna (reference).**

- Transmitted signal: s = x, with no scaling.
- Received and decision statistic: r = y = h x + w.
- Equivalent channel: gain h, noise variance 1/SNR.
- Post-detection SNR: \|h\|² · SNR. This is the fading channel of Problem 1.

**Transmit beamforming [2, Sec. 4.1].** The transmitter knows (h₁, h₂) and
sends the same symbol from both antennas with weights

  v = (h₁*, h₂*)ᵀ / ‖h‖,  that is s_l = v_l x = (h_l* / ‖h‖) x.

The conjugate phases cancel the channel phases, so the two components add
coherently at the receiver:

  y = (h₁v₁ + h₂v₂) x + w = ‖h‖ x + w.

- Decision statistic: r = y, with gain g = ‖h‖ and noise variance 1/SNR.
- Post-detection SNR: ‖h‖² · SNR = (\|h₁\|² + \|h₂\|²) · SNR.
- Scaling: the 1/‖h‖ factor makes ‖v‖ = 1, so the total energy per channel use
  is \|s₁\|² + \|s₂\|² = ‖v‖² \|x\|² = \|x\|², with mean one. On average each
  antenna carries E[\|h_l\|²/‖h‖²] = 1/2.

**Alamouti [2, Sec. 4.2].** The transmitter has no channel knowledge. Two
symbols x₁, x₂ are sent in two channel uses. Rows are channel uses and columns
are antennas:

  S = (1/√2) · [  x₁    x₂  ]   (channel use 1)
               [ −x₂*   x₁* ]   (channel use 2)

With h constant over the block, the received samples are

  y₁ = (h₁x₁ + h₂x₂)/√2 + w₁,  y₂ = (−h₁x₂* + h₂x₁*)/√2 + w₂.

Alamouti combining (normalised by ‖h‖) gives

  r₁ = (h₁* y₁ + h₂ y₂*) / ‖h‖ = (‖h‖/√2) x₁ + n₁,
  r₂ = (h₂* y₁ − h₁ y₂*) / ‖h‖ = (‖h‖/√2) x₂ + n₂.

- The orthogonal code cancels the cross-terms. n₁ = (h₁*w₁ + h₂w₂*)/‖h‖ and
  n₂ = (h₂*w₁ − h₁w₂*)/‖h‖ are each CN(0, 1/SNR) and uncorrelated
  (E[n₁n₂*] = (h₁*h₂ − h₂h₁*)/(‖h‖² SNR) = 0). Independent symbol-by-symbol ML
  detection is therefore optimal.
- Equivalent channel: gain g = ‖h‖/√2 for each symbol, noise variance 1/SNR.
- Post-detection SNR: ‖h‖² · SNR / 2.
- Scaling: the 1/√2 factor splits each channel use's energy over the two
  antennas, giving (\|x₁\|² + \|x₂\|²)/2 per channel use, with mean one.

**Numerical check of the energy constraint.** The simulation measures the
average transmitted energy per channel use, Σ_l E\|s_l\|², from the signals
actually sent at every SNR point (`summary.txt`):

| scheme | scaling | total energy per channel use (min / max over SNR points) | per antenna |
|---|---|---|---|
| single antenna | s = x | 1.000000 / 1.000000 | 1.0000 |
| beamforming | s_l = (h_l*/‖h‖) x | 1.000000 / 1.000000 | 0.5000 / 0.5000 |
| Alamouti | s_l = ±x_k^(*) / √2 | 1.000000 / 1.000000 | 0.5000 / 0.5000 |

All three schemes, and MRC (one transmit antenna sending x), transmit exactly
the energy of the single-antenna system, so the comparison is fair. The total
is 1 to machine precision, not just on average, because 4-QAM has constant
modulus (\|x\|² = 1).

**Consistency checks.**

- The single-antenna curve agrees with the exact 4-QAM Rayleigh SER of
  Problem 1 to within 1.9σ.
- Beamforming agrees with the independent MRC (L = 2) simulation to within
  1.5σ, as expected from its equivalent channel (below).

## Diversity order from the high-SNR slope

Local slopes −Δlog₁₀ SER / Δlog₁₀ SNR over the last 2 dB steps (`summary.txt`):

| scheme | SNR points (dB) | local slopes | predicted diversity |
|---|---|---|---|
| single antenna | 18–24 | 1.02 → 0.95 → 0.94 | 1 |
| beamforming | 14–20 | 1.90 → 1.77 → 2.26 | 2 |
| Alamouti | 18–24 | 1.94 → 1.99 → 2.61 | 2 |
| MRC, L = 2 | 16–22 | 1.83 → 2.10 → 2.04 | 2 |

The single antenna has diversity order 1. Beamforming, Alamouti and MRC all
have diversity order 2.

The last-step values deviate by up to ±0.3 from 2. That is the Monte Carlo
accuracy of a single 2 dB step with 50–200 errors at each end: about
0.43·√(1/n₁ + 1/n₂)/0.2 ≈ 0.2–0.3.

The reason is the deep-fade argument. In all three two-antenna schemes the
post-detection SNR is proportional to ‖h‖² = \|h₁\|² + \|h₂\|². An error is
therefore likely only when *both* paths fade at the same time, which has
probability Pr[‖h‖² < 1/SNR] ≈ SNR⁻²/2. Alamouti reaches full diversity without
any channel knowledge at the transmitter.

## SNR gaps and their origin

Horizontal gaps read off the simulated curves (`summary.txt`):

| pair | SER 10⁻² | SER 10⁻³ | SER 10⁻⁴ | predicted |
|---|---|---|---|---|
| Alamouti − beamforming | 3.00 dB | 3.01 dB | 2.78 dB | 10 log₁₀ 2 = 3.01 dB |
| Alamouti − MRC (L = 2) | 2.98 dB | 3.03 dB | 2.60 dB | 3.01 dB |
| beamforming − MRC (L = 2) | −0.02 dB | 0.02 dB | −0.18 dB | 0 dB |
| single antenna − Alamouti | 6.32 dB | – | – | grows as SER falls (slope 1 vs 2) |

The 10⁻⁴ column is the least accurate, about ±0.3 dB. There the beamforming and
MRC curves are interpolated between points with only 44–216 errors. All
measured gaps agree with the predictions to within this accuracy.

**Beamforming = MRC (L = 2), gap 0 dB.** Both reduce to the same equivalent
channel, r = ‖h‖ x + n with n ~ CN(0, 1/SNR) and ‖h‖² ~ Gamma(2, 1).

- Beamforming co-phases and weights at the transmitter (maximum ratio
  transmission). MRC does the same at the receiver.
- Both achieve diversity 2 and array gain E‖h‖² = 2, that is 3.01 dB. The
  array gains measured in the simulation are 3.01 dB for beamforming, 0.00 dB
  for Alamouti and 0.00 dB for the single antenna, matching theory.
- The difference lies in what each needs. MRC needs two receive chains.
  Beamforming needs channel knowledge at the transmitter, through feedback or
  reciprocity.

**Alamouti versus beamforming and MRC, gap 3 dB.** Alamouti achieves the same
diversity order (2) but no array gain.

- Without channel knowledge, the transmitter cannot steer its fixed energy so
  that the two signals add coherently. It has to split the energy equally (the
  1/√2 scaling), and the post-detection SNR is ‖h‖² · SNR/2. Its average,
  E‖h‖²/2 = 1, is the same as for a single antenna: array gain 0 dB.
- With beamforming, the same total energy arrives in phase, giving
  ‖h‖² · SNR and array gain 2.
- With MRC, the receiver collects the full transmitted energy on *each* of its
  two antennas, with independent noise, which doubles the received signal
  energy.
- The Alamouti and beamforming post-detection SNRs have identical
  distributions up to the factor 2, so SER_Alamouti(SNR) = SER_BF(SNR/2)
  exactly. The gap is therefore 10 log₁₀ 2 = 3.01 dB at every SER, not only
  asymptotically. The measured 3.00 dB and 3.01 dB at 10⁻² and 10⁻³ confirm it.

**Single antenna versus Alamouti.** This is a *diversity* gain, not an array
gain. Both have array gain 0 dB, but the slopes are 1 and 2. The gap therefore
widens without bound as the target SER decreases. It is 6.3 dB at 10⁻², and in
the simulated range the single-antenna curve does not even reach 10⁻³, while
Alamouti falls below 10⁻⁴.

**Summary.** Two transmit antennas give diversity order 2 with or without
channel knowledge at the transmitter. Channel knowledge (beamforming) adds a
3 dB array gain and matches two-antenna receive diversity (MRC). Alamouti
gives up that 3 dB in exchange for needing no feedback, while keeping full
rate (two symbols in two channel uses) and simple linear decoding.
