"""Unit-energy signal constellations (lecture notes [1, Sec. 2.2]).

Every constellation is a 1-D complex array X = [u_1, ..., u_M] with average
symbol energy E[|x|^2] = (1/M) sum |u_i|^2 = 1, so that the SNR is set by the
noise variance alone (G.1, G.2).
"""
import numpy as np


def normalize_energy(points):
    """Scale `points` so that their average energy is one."""
    points = np.asarray(points, dtype=complex)
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def bpsk():
    """BPSK: X = {-A, +A}, with A = 1 after normalisation."""
    return normalize_energy([-1, 1])


def qam4():
    """4-QAM (QPSK): X = {+-A +- jA}, with A = 1/sqrt(2) after normalisation."""
    return normalize_energy([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
