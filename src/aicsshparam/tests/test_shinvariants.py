"""Tests for aicsshparam.shinvariants.

Covers:
- Feature count and naming for lmax=5
- Power spectrum correctness (sum-of-squares definition)
- Bispectrum feature dtype (must be real-valued)
- CG coefficient known values and selection rules
- Rotation invariance of power spectrum and bispectrum (most critical)
- Dict vs DataFrame input paths for get_invariants
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pandas as pd
import pytest

from aicsshparam.shinvariants import (
    _bispectrum_from_array,
    _cg,
    _parse_coeffs_to_array,
    _power_spectrum_from_array,
    _valid_triples,
    bispectrum,
    get_invariants,
    power_spectrum,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toy_df(n: int, lmax: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create a DataFrame with random shcoeffs columns (aics-shparam format)."""
    data = {}
    for l in range(lmax + 1):
        data[f"shcoeffs_L{l}M0C"] = rng.standard_normal(n)
        for m in range(1, lmax + 1):
            data[f"shcoeffs_L{l}M{m}C"] = rng.standard_normal(n)
            data[f"shcoeffs_L{l}M{m}S"] = rng.standard_normal(n)
    return pd.DataFrame(data)


def _make_toy_dict(lmax: int, rng: np.random.Generator) -> dict:
    """Create a single-shape coefficient dict (aics-shparam format)."""
    data = {}
    for l in range(lmax + 1):
        data[f"shcoeffs_L{l}M0C"] = float(rng.standard_normal())
        for m in range(1, lmax + 1):
            data[f"shcoeffs_L{l}M{m}C"] = float(rng.standard_normal())
            data[f"shcoeffs_L{l}M{m}S"] = float(rng.standard_normal())
    return data


def _apply_wigner_d_rotation(f_lm: np.ndarray, lmax: int, rotation) -> np.ndarray:
    """Rotate complex SH coefficients using Wigner D-matrices.

    Parameters
    ----------
    f_lm : ndarray, shape ``(n, lmax+1, 2*lmax+1)``
    lmax : int
    rotation : scipy.spatial.transform.Rotation

    Returns
    -------
    f_rot : ndarray, same shape
    """
    euler = rotation.as_euler("ZYZ")
    alpha, beta, gamma = euler

    f_rot = np.zeros_like(f_lm)
    for l in range(lmax + 1):
        D = _wigner_d_matrix(l, alpha, beta, gamma)
        sl = slice(lmax - l, lmax + l + 1)
        f_rot[:, l, sl] = f_lm[:, l, sl] @ D.T

    return f_rot


def _wigner_d_matrix(l: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Wigner D-matrix D^l_{m'm}(alpha,beta,gamma) in ZYZ convention.

    ``D^l_{m'm}`` = ``e^{-i*m'*alpha}`` * ``d^l_{m'm}(beta)``
    * ``e^{-i*m*gamma}``
    """
    dim = 2 * l + 1
    ms = np.arange(-l, l + 1)
    D = np.zeros((dim, dim), dtype=complex)
    for mp_idx, mp in enumerate(ms):
        for m_idx, m in enumerate(ms):
            d_val = _small_d(l, mp, m, beta)
            D[mp_idx, m_idx] = (
                cmath.exp(-1j * mp * alpha) * d_val * cmath.exp(-1j * m * gamma)
            )
    return D


def _small_d(l: int, mp: int, m: int, beta: float) -> float:
    """Small Wigner d-matrix element ``d^l_{mp,m}(beta)``."""
    cos_b2 = math.cos(beta / 2)
    sin_b2 = math.sin(beta / 2)

    s_min = max(0, mp - m)
    s_max = min(l + mp, l - m)

    total = 0.0
    for s in range(s_min, s_max + 1):
        sign = (-1) ** (m - mp + s)
        try:
            coeff = math.sqrt(
                math.factorial(l + mp)
                * math.factorial(l - mp)
                * math.factorial(l + m)
                * math.factorial(l - m)
            ) / (
                math.factorial(l + mp - s)
                * math.factorial(s)
                * math.factorial(m - mp + s)
                * math.factorial(l - m - s)
            )
        except (ValueError, OverflowError):
            coeff = 0.0

        power_cos = 2 * l + mp - m - 2 * s
        power_sin = m - mp + 2 * s

        if power_cos < 0 or power_sin < 0:
            continue

        total += sign * coeff * (cos_b2**power_cos) * (sin_b2**power_sin)

    return total


# ---------------------------------------------------------------------------
# Tests: feature counts and naming
# ---------------------------------------------------------------------------


class TestFeatureCount:
    def test_lmax5_shape(self):
        rng = np.random.default_rng(0)
        df = _make_toy_df(n=7, lmax=5, rng=rng)
        X, names = get_invariants(df, lmax=5, include_bispectrum=True)
        assert X.shape == (7, 75), f"Expected (7, 75), got {X.shape}"
        assert len(names) == 75

    def test_power_only_shape(self):
        rng = np.random.default_rng(1)
        df = _make_toy_df(n=3, lmax=5, rng=rng)
        X, names = get_invariants(df, lmax=5, include_bispectrum=False)
        assert X.shape == (3, 6)
        assert len(names) == 6

    def test_bispectrum_triple_count_lmax5(self):
        triples = _valid_triples(lmax=5)
        assert len(triples) == 69

    def test_feature_names_prefix(self):
        rng = np.random.default_rng(2)
        df = _make_toy_df(n=2, lmax=5, rng=rng)
        _, names = get_invariants(df, lmax=5)
        assert names[0] == "power_l0"
        assert names[5] == "power_l5"
        assert names[6].startswith("bispec_")
        assert names[-1].startswith("bispec_")


# ---------------------------------------------------------------------------
# Tests: dict vs DataFrame input paths
# ---------------------------------------------------------------------------


class TestInputFormats:
    def test_dict_input_returns_1d(self):
        rng = np.random.default_rng(10)
        d = _make_toy_dict(lmax=5, rng=rng)
        X, names = get_invariants(d, lmax=5)
        assert X.ndim == 1
        assert X.shape == (75,)
        assert len(names) == 75

    def test_dict_and_df_consistent(self):
        """Single-row DataFrame and equivalent dict should give identical results."""
        rng = np.random.default_rng(11)
        d = _make_toy_dict(lmax=3, rng=rng)
        df = pd.DataFrame({k: [v] for k, v in d.items()})

        X_dict, _ = get_invariants(d, lmax=3)
        X_df, _ = get_invariants(df, lmax=3)

        np.testing.assert_allclose(X_dict, X_df.squeeze(0), rtol=1e-12)

    def test_power_spectrum_dict_input(self):
        rng = np.random.default_rng(12)
        d = _make_toy_dict(lmax=2, rng=rng)
        S, names = power_spectrum(d, lmax=2)
        assert S.shape == (1, 3)
        assert names == ["power_l0", "power_l1", "power_l2"]

    def test_bispectrum_dict_input(self):
        rng = np.random.default_rng(13)
        d = _make_toy_dict(lmax=2, rng=rng)
        B, names = bispectrum(d, lmax=2)
        assert B.shape[0] == 1
        assert B.dtype == np.float64 or np.issubdtype(B.dtype, np.floating)


# ---------------------------------------------------------------------------
# Tests: power spectrum correctness
# ---------------------------------------------------------------------------


class TestPowerSpectrum:
    def test_sum_of_squares(self):
        """S_l must equal sum_m |f_{l,m}|^2 for all l."""
        rng = np.random.default_rng(42)
        n, lmax = 4, 3
        df = _make_toy_df(n=n, lmax=lmax, rng=rng)
        f_lm = _parse_coeffs_to_array(df, lmax)
        S = _power_spectrum_from_array(f_lm, lmax)

        for sample in range(n):
            for l in range(lmax + 1):
                expected = sum(
                    abs(f_lm[sample, l, lmax + m]) ** 2 for m in range(-l, l + 1)
                )
                np.testing.assert_allclose(S[sample, l], expected, rtol=1e-12)

    def test_l0_only_cosine(self):
        """For l=0 there is only m=0, so S_0 = c_{0,0}^2."""
        df = pd.DataFrame(
            {
                "shcoeffs_L0M0C": [3.0],
                "shcoeffs_L1M0C": [0.0],
                "shcoeffs_L1M1C": [0.0],
                "shcoeffs_L1M1S": [0.0],
            }
        )
        f_lm = _parse_coeffs_to_array(df, lmax=1)
        S = _power_spectrum_from_array(f_lm, lmax=1)
        np.testing.assert_allclose(S[0, 0], 9.0, rtol=1e-12)  # 3^2


# ---------------------------------------------------------------------------
# Tests: bispectrum dtype
# ---------------------------------------------------------------------------


class TestBispectrumDtype:
    def test_output_is_real(self):
        """Bispectrum array must have a real floating dtype."""
        rng = np.random.default_rng(7)
        n, lmax = 5, 4
        df = _make_toy_df(n=n, lmax=lmax, rng=rng)
        B, _ = bispectrum(df, lmax)
        assert np.issubdtype(B.dtype, np.floating)


# ---------------------------------------------------------------------------
# Tests: CG coefficients
# ---------------------------------------------------------------------------


class TestCGCoefficient:
    def test_selection_rule(self):
        """CG is 0 when m1+m2 != m."""
        assert _cg(1, 1, 1, 1, 2, 1) == 0.0

    def test_known_value(self):
        """CG(1,0; 1,0 | 0,0) = -1/sqrt(3)."""
        val = _cg(1, 0, 1, 0, 0, 0)
        np.testing.assert_allclose(val, -1.0 / math.sqrt(3), atol=1e-10)

    def test_l0_coupling(self):
        """CG(l,m; 0,0 | l,m) = 1 for all l, m."""
        for l in range(4):
            for m in range(-l, l + 1):
                val = _cg(l, m, 0, 0, l, m)
                np.testing.assert_allclose(
                    val,
                    1.0,
                    atol=1e-10,
                    err_msg=f"Failed for l={l}, m={m}",
                )


# ---------------------------------------------------------------------------
# Tests: rotation invariance (most critical)
# ---------------------------------------------------------------------------


class TestRotationInvariance:
    """Power spectrum and bispectrum must be invariant under SO(3) rotation."""

    @pytest.fixture
    def rotated_arrays(self):
        pytest.importorskip("scipy.spatial.transform")
        from scipy.spatial.transform import Rotation

        rng = np.random.default_rng(99)
        lmax = 3  # keep small for speed
        n = 2
        df = _make_toy_df(n=n, lmax=lmax, rng=rng)
        f_orig = _parse_coeffs_to_array(df, lmax)

        rot = Rotation.from_euler("ZYZ", [0.3, 0.7, 1.1])
        f_rot = _apply_wigner_d_rotation(f_orig, lmax, rot)
        return f_orig, f_rot, lmax

    def test_power_spectrum_invariant(self, rotated_arrays):
        f_orig, f_rot, lmax = rotated_arrays
        S_orig = _power_spectrum_from_array(f_orig, lmax)
        S_rot = _power_spectrum_from_array(f_rot, lmax)
        np.testing.assert_allclose(
            S_orig,
            S_rot,
            atol=1e-8,
            err_msg="Power spectrum not invariant under rotation",
        )

    def test_bispectrum_invariant(self, rotated_arrays):
        f_orig, f_rot, lmax = rotated_arrays
        B_orig = _bispectrum_from_array(f_orig, lmax)
        B_rot = _bispectrum_from_array(f_rot, lmax)
        np.testing.assert_allclose(
            B_orig,
            B_rot,
            atol=1e-6,
            err_msg="Bispectrum not invariant under rotation",
        )
