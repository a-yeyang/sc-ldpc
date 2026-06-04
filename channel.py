"""BPSK modulation, AWGN channel and LLR computation."""
from __future__ import annotations
import numpy as np

LARGE_LLR = 30.0   # LLR magnitude used for "known" (perfectly reliable) bits


def ebn0_to_sigma(ebn0_db: float, rate: float) -> float:
    """Noise std-dev for BPSK (unit symbol energy Es=1) at a given Eb/N0 [dB]."""
    ebn0 = 10.0 ** (ebn0_db / 10.0)
    esn0 = rate * ebn0          # Es/N0 = R * Eb/N0
    n0 = 1.0 / esn0             # Es = 1
    return np.sqrt(n0 / 2.0)


def bpsk(bits: np.ndarray) -> np.ndarray:
    """bit 0 -> +1, bit 1 -> -1."""
    return 1.0 - 2.0 * bits.astype(np.float64)


def awgn(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return x + sigma * rng.standard_normal(x.shape)


def llr_awgn(y: np.ndarray, sigma: float) -> np.ndarray:
    """Channel LLR for BPSK (0->+1): L = 2y/sigma^2  (>0 favours bit 0)."""
    return 2.0 * y / (sigma * sigma)
