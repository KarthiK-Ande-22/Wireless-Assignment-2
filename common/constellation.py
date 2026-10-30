"""Unit-energy BPSK and 4-QAM constellations."""
import numpy as np


def normalize_energy(points):
    """Scale the points so that the average symbol energy is 1."""
    points = np.asarray(points, dtype=complex)
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def bpsk():
    """BPSK: {-1, +1}."""
    return normalize_energy([-1, 1])


def qam4():
    """4-QAM: (+-1 +- j) / sqrt(2)."""
    return normalize_energy([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
