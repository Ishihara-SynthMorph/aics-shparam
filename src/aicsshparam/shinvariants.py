"""Rotation-invariant features from SPHARM coefficients.

Implements SO(3)-invariant descriptors (power spectrum and bispectrum)
from spherical harmonic coefficients produced by aics-shparam.
Both features are provably rotation-invariant, unlike raw SH coefficients.

References
----------
Kazhdan et al. (2003) "Rotation Invariant Spherical Harmonic Representation
of 3D Shape Descriptors".

Kondor (2007) "A novel set of rotationally and translationally invariant
features for images".
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd


@lru_cache(maxsize=None)
def _wigner3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """Wigner 3j symbol via the Racah formula.

    All arguments must be non-negative integers (not half-integers).

    Parameters
    ----------
    j1, j2, j3 : int
        Angular momentum quantum numbers satisfying the triangle inequality.
    m1, m2, m3 : int
        Magnetic quantum numbers satisfying ``|mi| <= ji``.

    Returns
    -------
    float
        The Wigner 3j symbol value.  Returns 0 when any selection rule is
        violated.
    """
    # Selection rule: m1 + m2 + m3 = 0
    if m1 + m2 + m3 != 0:
        return 0.0
    # Bounds on m
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    # Triangle inequality
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return 0.0

    def _triangle(a: int, b: int, c: int) -> float:
        return (
            math.factorial(a + b - c)
            * math.factorial(a - b + c)
            * math.factorial(-a + b + c)
            / math.factorial(a + b + c + 1)
        )

    tri = _triangle(j1, j2, j3)
    prefactor = (-1) ** (j1 - j2 - m3) * math.sqrt(
        tri
        * math.factorial(j1 + m1)
        * math.factorial(j1 - m1)
        * math.factorial(j2 + m2)
        * math.factorial(j2 - m2)
        * math.factorial(j3 + m3)
        * math.factorial(j3 - m3)
    )

    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    s = 0.0
    for t in range(t_min, t_max + 1):
        s += (-1) ** t / (
            math.factorial(t)
            * math.factorial(j1 + j2 - j3 - t)
            * math.factorial(j1 - m1 - t)
            * math.factorial(j2 + m2 - t)
            * math.factorial(j3 - j2 + m1 + t)
            * math.factorial(j3 - j1 - m2 + t)
        )

    return prefactor * s


@lru_cache(maxsize=None)
def _cg(l1: int, m1: int, l2: int, m2: int, l: int, m: int) -> float:  # noqa: E741
    """Clebsch-Gordan coefficient <l1,m1; l2,m2 | l,m>.

    Computed via the Wigner 3j symbol relation::

        CG(l1,m1; l2,m2 | l,m)
            = (-1)^(l1-l2+m) * sqrt(2l+1) * W3j(l1,l2,l; m1,m2,-m)

    Parameters
    ----------
    l1, m1, l2, m2, l, m : int
        Angular momentum quantum numbers satisfying
        ``|m1| <= l1``, ``|m2| <= l2``, ``|m| <= l``.

    Returns
    -------
    float
        The Clebsch-Gordan coefficient.  Returns 0 immediately when the
        selection rule ``m1 + m2 == m`` is violated.
    """
    if m1 + m2 != m:
        return 0.0
    return (
        (-1) ** (l1 - l2 + m) * math.sqrt(2 * l + 1) * _wigner3j(l1, l2, l, m1, m2, -m)
    )


def _parse_coeffs_to_array(coeffs, lmax: int) -> np.ndarray:
    """Parse aics-shparam SH coefficients into a complex array.

    Accepts a ``dict`` (single shape, as returned by ``get_shcoeffs``) or a
    ``pd.DataFrame`` (batch output with ``shcoeffs_L{l}M{m}C/S`` columns).

    Real-to-complex conversion (physics convention)::

        f_{l,0}  = c_{l,0}
        f_{l,+m} = (-1)^m / sqrt(2) * (c - i*s)   for m > 0
        f_{l,-m} = 1/sqrt(2) * (c + i*s)           for m > 0

    Parameters
    ----------
    coeffs : dict or pd.DataFrame
        SPHARM coefficients in aics-shparam format.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(n_samples, lmax+1, 2*lmax+1)`` where
        element ``[n, l, lmax+m]`` stores ``f_{l,m}``.
    """
    if isinstance(coeffs, dict):
        df = pd.DataFrame(
            {k: [v] for k, v in coeffs.items() if k.startswith("shcoeffs_")}
        )
    else:
        df = coeffs

    n = len(df)
    f = np.zeros((n, lmax + 1, 2 * lmax + 1), dtype=complex)

    for l in range(lmax + 1):  # noqa: E741
        col_c = f"shcoeffs_L{l}M0C"
        if col_c in df.columns:
            f[:, l, lmax] = df[col_c].values  # m=0 at index lmax

        for m in range(1, l + 1):
            col_c = f"shcoeffs_L{l}M{m}C"
            col_s = f"shcoeffs_L{l}M{m}S"
            if col_c not in df.columns or col_s not in df.columns:
                continue
            c = df[col_c].values
            s = df[col_s].values
            sign = (-1) ** m
            inv_sqrt2 = 1.0 / math.sqrt(2)
            f[:, l, lmax + m] = sign * inv_sqrt2 * (c - 1j * s)  # f_{l,+m}
            f[:, l, lmax - m] = inv_sqrt2 * (c + 1j * s)  # f_{l,-m}

    return f


def _valid_triples(lmax: int) -> list[tuple[int, int, int]]:
    """Return all valid (l1, l2, l) bispectrum index triples for a given lmax.

    A triple ``(l1, l2, l)`` is valid when ``0 <= l1 <= l2 <= lmax`` and
    ``|l1 - l2| <= l <= min(l1 + l2, lmax)``.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    list of tuple[int, int, int]
        Sorted list of valid triples.  For ``lmax=5`` there are 69 triples.
    """
    triples = []
    for l1 in range(lmax + 1):
        for l2 in range(l1, lmax + 1):
            for l in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):  # noqa: E741
                triples.append((l1, l2, l))
    return triples


def _power_spectrum_from_array(f_lm: np.ndarray, lmax: int) -> np.ndarray:
    """Compute power spectrum from a complex coefficient array.

    Parameters
    ----------
    f_lm : np.ndarray, shape ``(n_samples, lmax+1, 2*lmax+1)``
        Complex SH coefficients as returned by ``_parse_coeffs_to_array``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    np.ndarray, shape ``(n_samples, lmax+1)``
        Power spectrum values ``S_l = sum_m |f_{l,m}|^2``.
    """
    n = f_lm.shape[0]
    S = np.zeros((n, lmax + 1))
    for l in range(lmax + 1):  # noqa: E741
        m_slice = slice(lmax - l, lmax + l + 1)
        S[:, l] = np.sum(np.abs(f_lm[:, l, m_slice]) ** 2, axis=1)
    return S


def _bispectrum_from_array(f_lm: np.ndarray, lmax: int) -> np.ndarray:
    """Compute bispectrum from a complex coefficient array.

    Parameters
    ----------
    f_lm : np.ndarray, shape ``(n_samples, lmax+1, 2*lmax+1)``
        Complex SH coefficients as returned by ``_parse_coeffs_to_array``.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    np.ndarray, shape ``(n_samples, n_triples)``
        Real-valued bispectrum features.
        ``n_triples = 69`` for ``lmax = 5``.
    """
    triples = _valid_triples(lmax)
    n = f_lm.shape[0]
    B = np.zeros((n, len(triples)))

    for idx, (l1, l2, l) in enumerate(triples):  # noqa: E741
        # Precompute nonzero CG entries for this triple
        cg_entries = []
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m = m1 + m2
                if abs(m) > l:
                    continue
                cg_val = _cg(l1, m1, l2, m2, l, m)
                if cg_val != 0.0:
                    cg_entries.append((cg_val, m1, m2, m))

        # Sum over nonzero CG entries (vectorised over batch dimension)
        acc = np.zeros(n, dtype=complex)
        for cg_val, m1, m2, m in cg_entries:
            acc += cg_val * (
                f_lm[:, l1, lmax + m1]
                * f_lm[:, l2, lmax + m2]
                * np.conj(f_lm[:, l, lmax + m])
            )
        B[:, idx] = acc.real

    return B


def power_spectrum(coeffs, lmax: int) -> tuple[np.ndarray, list[str]]:
    """Compute rotation-invariant power spectrum from SPHARM coefficients.

    The power spectrum is defined as ``S_l = sum_m |f_{l,m}|^2``.

    Parameters
    ----------
    coeffs : dict or pd.DataFrame
        SPHARM coefficients in aics-shparam format (keys/columns
        ``shcoeffs_L{l}M{m}C`` and ``shcoeffs_L{l}M{m}S``).
        Pass a ``dict`` (single shape, as returned by ``get_shcoeffs``)
        or a ``pd.DataFrame`` for batch processing.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    S : np.ndarray, shape ``(n_samples, lmax+1)``
        Power spectrum values.
    names : list[str]
        Feature names, e.g. ``['power_l0', 'power_l1', ...]``.
    """
    f_lm = _parse_coeffs_to_array(coeffs, lmax)
    S = _power_spectrum_from_array(f_lm, lmax)
    names = [f"power_l{l}" for l in range(lmax + 1)]  # noqa: E741
    return S, names


def bispectrum(coeffs, lmax: int) -> tuple[np.ndarray, list[str]]:
    """Compute rotation-invariant bispectrum from SPHARM coefficients.

    The bispectrum coupling ``(l1, l2, l)`` is defined as::

        B_{l1,l2,l} = Re( sum_{m1,m2}
            CG(l1,m1; l2,m2 | l,m1+m2)
            * f_{l1,m1} * f_{l2,m2} * conj(f_{l,m1+m2}) )

    Parameters
    ----------
    coeffs : dict or pd.DataFrame
        SPHARM coefficients in aics-shparam format.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    B : np.ndarray, shape ``(n_samples, n_triples)``
        Bispectrum values.  ``n_triples = 69`` for ``lmax = 5``.
    names : list[str]
        Feature names, e.g. ``['bispec_0_0_0', ...]``.
    """
    f_lm = _parse_coeffs_to_array(coeffs, lmax)
    B = _bispectrum_from_array(f_lm, lmax)
    names = [f"bispec_{l1}_{l2}_{l}" for l1, l2, l in _valid_triples(lmax)]  # noqa: E741
    return B, names


def get_invariants(
    coeffs,
    lmax: int,
    include_bispectrum: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Compute rotation-invariant SPHARM features.

    Concatenates power spectrum and (optionally) bispectrum features.
    Both are provably SO(3)-invariant under rotations of the underlying shape.

    Parameters
    ----------
    coeffs : dict or pd.DataFrame
        SPHARM coefficients in aics-shparam format (keys/columns
        ``shcoeffs_L{l}M{m}C`` and ``shcoeffs_L{l}M{m}S``).
        Passing a ``dict`` (single shape, as returned by ``get_shcoeffs``)
        returns a 1-D feature vector; passing a ``pd.DataFrame`` returns a
        2-D array with one row per shape.
    lmax : int
        Maximum spherical harmonic degree.
    include_bispectrum : bool, optional
        If ``True`` (default), append bispectrum features after the power
        spectrum.  For ``lmax=5`` this gives 6 + 69 = 75 features total.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_features,)`` for dict input or
        ``(n_samples, n_features)`` for DataFrame input.
    feature_names : list[str]
        Names for each element / column of X.

    Examples
    --------
    Single-shape dict input (from ``get_shcoeffs``):

    >>> coeffs = {"shcoeffs_L0M0C": 1.0, "shcoeffs_L1M0C": 0.5, ...}
    >>> X, names = get_invariants(coeffs, lmax=5)
    >>> X.shape
    (75,)

    Batch DataFrame input:

    >>> X, names = get_invariants(df, lmax=5)
    >>> X.shape
    (n_samples, 75)
    """
    is_dict = isinstance(coeffs, dict)
    f_lm = _parse_coeffs_to_array(coeffs, lmax)

    S = _power_spectrum_from_array(f_lm, lmax)
    ps_names = [f"power_l{l}" for l in range(lmax + 1)]  # noqa: E741

    if include_bispectrum:
        B = _bispectrum_from_array(f_lm, lmax)
        bs_names = [f"bispec_{l1}_{l2}_{l}" for l1, l2, l in _valid_triples(lmax)]  # noqa: E741
        X = np.concatenate([S, B], axis=1)
        feature_names = ps_names + bs_names
    else:
        X = S
        feature_names = ps_names

    if is_dict:
        X = X.squeeze(0)

    return X, feature_names
