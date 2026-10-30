"""Receive combiners.

Each combiner turns (y, h) into one scalar channel r = g x + n with
n ~ CN(0, 1/SNR), so the same ML detector can be used for every receiver.
"""
import numpy as np


def no_combining(y, h):
    """Single antenna: y = h x + w is already a scalar channel."""
    return y, h


def mrc_combine(y, h):
    """MRC: r = h^H y / ||h|| = ||h|| x + n, so the gain is ||h||."""
    h_norm = np.linalg.norm(h, axis=-1)
    r = np.sum(np.conj(h) * y, axis=-1) / h_norm
    return r, h_norm


def selection_combine(y, h):
    """SC: keep only the antenna with the largest |h_l|^2, so the gain is that h_l."""
    best = np.argmax(np.abs(h) ** 2, axis=-1)
    rows = np.arange(h.shape[0])
    return y[rows, best], h[rows, best]
