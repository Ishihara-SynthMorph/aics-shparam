"""Rotation-invariant features from SPHARM coefficients.

Implements two families of invariant descriptors from spherical harmonic
coefficients produced by aics-shparam:

* **SO(3) invariants** (``power_spectrum``, ``bispectrum``,
  ``get_invariants``) -- invariant under *any* 3D rotation.
* **SO(2) invariants** (``so2_power_spectrum``, ``so2_cross_power``,
  ``so2_bispectrum``, ``get_so2_invariants``) -- invariant only under
  rotation about the z-axis, while *retaining* the polar (up/down)
  information that SO(3) averages away.  Useful when there is a physically
  meaningful axis (gravity, apical-basal polarity, optical axis).

Both families are provably invariant under their respective symmetry group,
unlike raw SH coefficients.

The SO(2) case is abelian: under a z-rotation by ``alpha`` each coefficient
picks up a pure phase ``f_{l,m} -> exp(-i*m*alpha) * f_{l,m}``.  Any product
of coefficients whose signed ``m`` values sum to zero is therefore invariant,
and -- unlike the SO(3) case -- no Clebsch-Gordan / Wigner-3j weights are
needed.

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
    names = [
        f"bispec_{l1}_{l2}_{l}" for l1, l2, l in _valid_triples(lmax)
    ]  # noqa: E741
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
        bs_names = [
            f"bispec_{l1}_{l2}_{l}" for l1, l2, l in _valid_triples(lmax)
        ]  # noqa: E741
        X = np.concatenate([S, B], axis=1)
        feature_names = ps_names + bs_names
    else:
        X = S
        feature_names = ps_names

    if is_dict:
        X = X.squeeze(0)

    return X, feature_names


# ---------------------------------------------------------------------------
# SO(2) invariants (rotation about the z-axis only)
# ---------------------------------------------------------------------------
#
# Under a z-rotation by alpha, f_{l,m} -> exp(-i*m*alpha) * f_{l,m}.  Because
# the group is abelian (1-D irreps), any product of coefficients whose signed
# m's sum to zero is invariant -- no Clebsch-Gordan weights required.  These
# features keep each m separately (rather than summing over m as the SO(3)
# power spectrum does), so they retain polar information but are tied to the
# chosen z-axis.
#
# The aics-shparam coefficients describe a real-valued function, so
# f_{l,-m} = (-1)^m * conj(f_{l,m}).  Consequently |f_{l,-m}| = |f_{l,m}| and
# the m<0 features are redundant with the m>=0 ones; the enumerations below
# use this to avoid duplicate features.


def _so2_power_from_array(f_lm: np.ndarray, lmax: int):
    """Per-(l, m) power ``|f_{l,m}|^2`` for ``m = 0 .. l``.

    Each term is z-rotation invariant because ``|exp(-i*m*alpha)| = 1``.
    Only ``m >= 0`` is kept since ``|f_{l,-m}| = |f_{l,m}|`` for a real field.

    Returns
    -------
    X : np.ndarray, shape ``(n_samples, n_features)``
    names : list[str]
    """
    n = f_lm.shape[0]
    feats = []
    names = []
    for l in range(lmax + 1):  # noqa: E741
        for m in range(l + 1):
            feats.append(np.abs(f_lm[:, l, lmax + m]) ** 2)
            names.append(f"so2_power_l{l}_m{m}")
    X = np.stack(feats, axis=1) if feats else np.zeros((n, 0))
    return X, names


def _so2_cross_from_array(f_lm: np.ndarray, lmax: int):
    """Same-m cross-degree correlations ``f_{l1,m} * conj(f_{l2,m})``.

    For each shared order ``m`` and pair of degrees ``l1 < l2`` (both
    ``>= m``), the product's phases ``exp(-i*m*alpha)`` and
    ``exp(+i*m*alpha)`` cancel, so both real and imaginary parts are
    invariant.  The imaginary part encodes the relative azimuthal offset
    between degrees and vanishes identically for ``m = 0`` (real
    coefficients), so it is emitted only for ``m >= 1``.

    Returns
    -------
    X : np.ndarray, shape ``(n_samples, n_features)``
    names : list[str]
    """
    n = f_lm.shape[0]
    feats = []
    names = []
    for m in range(lmax + 1):
        degrees = list(range(m, lmax + 1))
        for i in range(len(degrees)):
            for j in range(i + 1, len(degrees)):
                l1, l2 = degrees[i], degrees[j]
                prod = f_lm[:, l1, lmax + m] * np.conj(f_lm[:, l2, lmax + m])
                feats.append(prod.real)
                names.append(f"so2_cross_l{l1}_l{l2}_m{m}_re")
                if m >= 1:
                    feats.append(prod.imag)
                    names.append(f"so2_cross_l{l1}_l{l2}_m{m}_im")
    X = np.stack(feats, axis=1) if feats else np.zeros((n, 0))
    return X, names


def _so2_bispectrum_terms(lmax: int):
    """Canonical, de-duplicated SO(2) bispectrum terms.

    A term couples ``f_{l1,m1} * f_{l2,m2} * conj(f_{l3,m3})`` with
    ``m3 = m1 + m2`` (so the signed m's sum to zero -> invariant).  Two
    symmetries are quotiented out to avoid redundant features:

    * **Swap** ``(l1, m1) <-> (l2, m2)`` leaves the product unchanged.
    * **Sign flip** ``(m1, m2, m3) -> (-m1, -m2, -m3)`` maps the product to
      its complex conjugate (real field identity), duplicating Re and
      negating Im.

    Each orbit under these symmetries contributes one term.  ``has_imag`` is
    ``False`` when the term is fixed by the sign flip (the product is real,
    e.g. all m = 0), in which case the imaginary part is identically zero.

    Returns
    -------
    list of tuple ``(l1, m1, l2, m2, l3, has_imag)``
    """
    seen = set()
    terms = []
    for l1 in range(lmax + 1):  # noqa: E741
        for l2 in range(l1, lmax + 1):  # noqa: E741
            for m1 in range(-l1, l1 + 1):
                for m2 in range(-l2, l2 + 1):
                    m3 = m1 + m2
                    if abs(m3) > lmax:
                        continue
                    # canonical key under swap (sorted pair) and sign flip
                    key_pos = (tuple(sorted([(l1, m1), (l2, m2)])),)
                    key_neg = (tuple(sorted([(l1, -m1), (l2, -m2)])),)
                    has_imag = key_pos != key_neg
                    for l3 in range(abs(m3), lmax + 1):  # noqa: E741
                        canon = min(
                            (key_pos[0], l3),
                            (key_neg[0], l3),
                        )
                        if canon in seen:
                            continue
                        seen.add(canon)
                        terms.append((l1, m1, l2, m2, l3, has_imag))
    return terms


def _so2_bispectrum_from_array(f_lm: np.ndarray, lmax: int):
    """SO(2) bispectrum ``f_{l1,m1} * f_{l2,m2} * conj(f_{l3,m1+m2})``.

    Fixes the relative phases between different m-channels, which the
    same-m cross terms cannot capture.

    Returns
    -------
    X : np.ndarray, shape ``(n_samples, n_features)``
    names : list[str]
    """
    n = f_lm.shape[0]
    feats = []
    names = []
    for l1, m1, l2, m2, l3, has_imag in _so2_bispectrum_terms(lmax):
        m3 = m1 + m2
        prod = (
            f_lm[:, l1, lmax + m1]
            * f_lm[:, l2, lmax + m2]
            * np.conj(f_lm[:, l3, lmax + m3])
        )
        base = f"so2_bispec_l{l1}m{m1}_l{l2}m{m2}_l{l3}"
        feats.append(prod.real)
        names.append(f"{base}_re")
        if has_imag:
            feats.append(prod.imag)
            names.append(f"{base}_im")
    X = np.stack(feats, axis=1) if feats else np.zeros((n, 0))
    return X, names


def so2_power_spectrum(coeffs, lmax: int) -> tuple[np.ndarray, list[str]]:
    """Compute the SO(2) per-(l, m) power spectrum.

    Each feature is ``|f_{l,m}|^2`` for ``m = 0 .. l``, invariant under
    rotation about the z-axis.  Unlike the SO(3) power spectrum this keeps
    each ``m`` separately, retaining the polar distribution of the shape.

    Parameters
    ----------
    coeffs : dict or pd.DataFrame
        SPHARM coefficients in aics-shparam format.  Pass a ``dict`` for a
        single shape or a ``pd.DataFrame`` for batch processing.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_features,)`` for dict input or
        ``(n_samples, n_features)`` for DataFrame input.
    names : list[str]
        Feature names, e.g. ``['so2_power_l0_m0', 'so2_power_l1_m0', ...]``.
    """
    is_dict = isinstance(coeffs, dict)
    f_lm = _parse_coeffs_to_array(coeffs, lmax)
    X, names = _so2_power_from_array(f_lm, lmax)
    if is_dict:
        X = X.squeeze(0)
    return X, names


def so2_cross_power(coeffs, lmax: int) -> tuple[np.ndarray, list[str]]:
    """Compute SO(2) same-m cross-degree correlation features.

    Real and imaginary parts of ``f_{l1,m} * conj(f_{l2,m})`` for ``l1 < l2``
    sharing order ``m``.  The imaginary part (``m >= 1`` only) encodes the
    relative azimuthal offset between degrees.

    Parameters
    ----------
    coeffs : dict or pd.DataFrame
        SPHARM coefficients in aics-shparam format.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_features,)`` for dict input or
        ``(n_samples, n_features)`` for DataFrame input.
    names : list[str]
        Feature names, e.g. ``['so2_cross_l0_l1_m0_re', ...]``.
    """
    is_dict = isinstance(coeffs, dict)
    f_lm = _parse_coeffs_to_array(coeffs, lmax)
    X, names = _so2_cross_from_array(f_lm, lmax)
    if is_dict:
        X = X.squeeze(0)
    return X, names


def so2_bispectrum(coeffs, lmax: int) -> tuple[np.ndarray, list[str]]:
    """Compute the SO(2) bispectrum.

    Real and imaginary parts of ``f_{l1,m1} * f_{l2,m2} * conj(f_{l3,m1+m2})``
    over a canonical, de-duplicated set of couplings.  These fix the relative
    phases between different m-channels.

    Note
    ----
    The number of bispectrum features grows quickly with ``lmax``; this tier
    is intended as an optional, higher-order complement to the power and
    cross-power features.

    Parameters
    ----------
    coeffs : dict or pd.DataFrame
        SPHARM coefficients in aics-shparam format.
    lmax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    X : np.ndarray
        Shape ``(n_features,)`` for dict input or
        ``(n_samples, n_features)`` for DataFrame input.
    names : list[str]
        Feature names, e.g. ``['so2_bispec_l1m1_l1m-1_l0_re', ...]``.
    """
    is_dict = isinstance(coeffs, dict)
    f_lm = _parse_coeffs_to_array(coeffs, lmax)
    X, names = _so2_bispectrum_from_array(f_lm, lmax)
    if is_dict:
        X = X.squeeze(0)
    return X, names


def get_so2_invariants(
    coeffs,
    lmax: int,
    include_cross: bool = True,
    include_bispectrum: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Compute SO(2) (z-axis) rotation-invariant SPHARM features.

    Concatenates the per-(l, m) power spectrum with, optionally, the same-m
    cross-degree correlations and the SO(2) bispectrum.  All features are
    invariant under rotation of the shape about the z-axis but -- unlike the
    SO(3) features -- retain the polar (up/down) distribution.

    The three tiers, in increasing order, are:

    1. ``so2_power``  -- ``|f_{l,m}|^2`` (always included).
    2. ``so2_cross``  -- same-m cross-degree correlations
       (``include_cross``, default ``True``).
    3. ``so2_bispec`` -- SO(2) bispectrum
       (``include_bispectrum``, default ``False``; grows quickly with
       ``lmax``).

    Parameters
    ----------
    coeffs : dict or pd.DataFrame
        SPHARM coefficients in aics-shparam format (keys/columns
        ``shcoeffs_L{l}M{m}C`` and ``shcoeffs_L{l}M{m}S``).
        Passing a ``dict`` returns a 1-D feature vector; passing a
        ``pd.DataFrame`` returns a 2-D array with one row per shape.
    lmax : int
        Maximum spherical harmonic degree.
    include_cross : bool, optional
        Append same-m cross-degree features (default ``True``).
    include_bispectrum : bool, optional
        Append SO(2) bispectrum features (default ``False``).

    Returns
    -------
    X : np.ndarray
        Shape ``(n_features,)`` for dict input or
        ``(n_samples, n_features)`` for DataFrame input.
    feature_names : list[str]
        Names for each element / column of X.

    Notes
    -----
    SO(2) invariants are tied to the chosen z-axis and are only meaningful if
    that axis is defined consistently across samples.  aics-shparam computes
    coefficients in the fixed image frame, so the z-axis is the image z-axis.

    Examples
    --------
    >>> X, names = get_so2_invariants(coeffs, lmax=5)
    >>> X, names = get_so2_invariants(df, lmax=5, include_bispectrum=True)
    """
    is_dict = isinstance(coeffs, dict)
    f_lm = _parse_coeffs_to_array(coeffs, lmax)

    blocks = []
    feature_names: list[str] = []

    Xp, names_p = _so2_power_from_array(f_lm, lmax)
    blocks.append(Xp)
    feature_names += names_p

    if include_cross:
        Xc, names_c = _so2_cross_from_array(f_lm, lmax)
        blocks.append(Xc)
        feature_names += names_c

    if include_bispectrum:
        Xb, names_b = _so2_bispectrum_from_array(f_lm, lmax)
        blocks.append(Xb)
        feature_names += names_b

    X = np.concatenate(blocks, axis=1)

    if is_dict:
        X = X.squeeze(0)

    return X, feature_names
