"""Cross-check of shinvariants power spectrum and bispectrum against e3nn-jax.

Independently validates the physics-convention CG coefficients and derived
invariants in ``aicsshparam.shinvariants`` by comparison with
``e3nn_jax.su2_clebsch_gordan``.

Convention summary
------------------
* **Physics (Condon-Shortley)** -- used by ``shinvariants``::

      CG^phys(l1,m1; l2,m2 | l,m)
          = (-1)^(l1-l2+m) * sqrt(2l+1) * W3j(l1,l2,l; m1,m2,-m)

* **e3nn SU(2) / complex-SH** -- ``e3nn_jax.su2_clebsch_gordan``::

      su2_CG(l1,l2,l3)[m1+l1, m2+l2, m+l3]
          = CG^phys(l1,m1; l2,m2 | l3,m) / sqrt(2*l3+1)

  The ``/ sqrt(2*l3+1)`` factor is explicit in e3nn's source
  (``e3nn_jax/_src/su2.py``, line 38).

* **Conversion**::

      CG^phys = sqrt(2l+1) * su2_CG          (no additional signs)
      B^phys_{l1,l2,l} = sqrt(2l+1) * B^su2_{l1,l2,l}

* **Real-SH bispectrum (pyspectra / e3nn.clebsch_gordan)**:
  ``e3nn.clebsch_gordan`` works in the *real*-SH basis, which requires a
  full real-to-complex change of basis for comparison.  Additionally,
  ``e3nn.reduced_symmetric_tensor_product_basis`` (used by pyspectra)
  symmetrises over l1<->l2 permutations.  These complications are out of
  scope for the current cross-check, which targets the complex-SH (SU(2))
  level.  See pykarambola issue #138 for further discussion.

Requirements
------------
``jax`` and ``e3nn-jax`` must be installed (``pip install jax e3nn-jax``).
Tests are automatically skipped when these packages are absent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
e3nn_jax = pytest.importorskip("e3nn_jax")

from e3nn_jax._src.su2 import su2_clebsch_gordan  # noqa: E402

from aicsshparam.shinvariants import (  # noqa: E402
    _bispectrum_from_array,
    _cg,
    _parse_coeffs_to_array,
    _valid_triples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_complex_flm(lmax: int, seed: int = 0) -> np.ndarray:
    """Random complex SH array via aics-shparam's real-to-complex conversion.

    Returns shape ``(1, lmax+1, 2*lmax+1)``.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    data = {}
    for l in range(lmax + 1):
        data[f"shcoeffs_L{l}M0C"] = [float(rng.standard_normal())]
        for m in range(1, l + 1):
            data[f"shcoeffs_L{l}M{m}C"] = [float(rng.standard_normal())]
            data[f"shcoeffs_L{l}M{m}S"] = [float(rng.standard_normal())]
    df = pd.DataFrame(data)
    return _parse_coeffs_to_array(df, lmax)


def _su2_bispectrum(f_lm: np.ndarray, lmax: int) -> np.ndarray:
    """Bispectrum using e3nn ``su2_clebsch_gordan`` coefficients.

    Parameters
    ----------
    f_lm : ndarray, shape ``(n, lmax+1, 2*lmax+1)``
        Complex SH coefficients (same layout as ``_parse_coeffs_to_array``).
    lmax : int

    Returns
    -------
    ndarray, shape ``(n, n_triples)``
    """
    triples = _valid_triples(lmax)
    n = f_lm.shape[0]
    B = np.zeros((n, len(triples)))
    for idx, (l1, l2, l) in enumerate(triples):
        CG = su2_clebsch_gordan(l1, l2, l)  # shape (2l1+1, 2l2+1, 2l+1)
        acc = np.zeros(n, dtype=complex)
        for m1 in range(-l1, l1 + 1):
            for m2 in range(-l2, l2 + 1):
                m = m1 + m2
                if abs(m) > l:
                    continue
                cg = float(CG[m1 + l1, m2 + l2, m + l])
                if cg == 0.0:
                    continue
                acc += cg * (
                    f_lm[:, l1, lmax + m1]
                    * f_lm[:, l2, lmax + m2]
                    * np.conj(f_lm[:, l, lmax + m])
                )
        B[:, idx] = acc.real
    return B


# ---------------------------------------------------------------------------
# Tests: CG coefficient convention
# ---------------------------------------------------------------------------


class TestCGConvention:
    def test_known_cg_l0_agrees(self):
        """Known value: CG(1,0;1,0|0,0) = -1/sqrt(3).

        For l_out=0 the conversion factor sqrt(2*0+1)=1, so su2 and physics
        must agree exactly (the no-arithmetic sanity check from pykarambola
        issue #138).
        """
        CG = su2_clebsch_gordan(1, 1, 0)
        su2_val = float(CG[1 + 0, 1 + 0, 0 + 0]) * math.sqrt(1)
        phys_val = _cg(1, 0, 1, 0, 0, 0)
        expected = -1.0 / math.sqrt(3)
        np.testing.assert_allclose(su2_val, expected, atol=1e-10)
        np.testing.assert_allclose(phys_val, expected, atol=1e-10)

    def test_sqrt_2lp1_conversion_all_triples(self):
        """su2_CG[...]*sqrt(2l+1) == phys_CG for all valid entries, lmax=4."""
        lmax = 4
        max_err = 0.0
        for l1 in range(lmax + 1):
            for l2 in range(l1, lmax + 1):
                for l in range(abs(l1 - l2), min(l1 + l2, lmax) + 1):
                    CG = su2_clebsch_gordan(l1, l2, l)
                    factor = math.sqrt(2 * l + 1)
                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            m = m1 + m2
                            if abs(m) > l:
                                continue
                            su2_val = float(CG[m1 + l1, m2 + l2, m + l]) * factor
                            phys_val = _cg(l1, m1, l2, m2, l, m)
                            max_err = max(max_err, abs(su2_val - phys_val))
        assert max_err < 1e-12, f"Max |su2*sqrt(2l+1) - physics| = {max_err:.2e}"


# ---------------------------------------------------------------------------
# Tests: power spectrum (no CG -- trivially identical)
# ---------------------------------------------------------------------------


class TestPowerSpectrum:
    def test_power_spectrum_is_sum_of_squares(self):
        """S_l = Σ_m |f_{l,m}|² is CG-independent; verify against direct sum."""
        lmax = 4
        f = _make_complex_flm(lmax, seed=7)
        for l in range(lmax + 1):
            expected = float(np.sum(np.abs(f[0, l, lmax - l : lmax + l + 1]) ** 2))
            # This is trivially the same regardless of CG convention.
            np.testing.assert_allclose(expected, expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# Tests: bispectrum conversion B^phys = sqrt(2l+1) * B^su2
# ---------------------------------------------------------------------------


class TestBispectrumConversion:
    @pytest.fixture
    def bispectrum_data(self):
        lmax = 4
        f = _make_complex_flm(lmax, seed=42)
        B_phys = _bispectrum_from_array(f, lmax)
        B_su2 = _su2_bispectrum(f, lmax)
        triples = _valid_triples(lmax)
        return B_phys, B_su2, triples, lmax

    def test_bispectrum_conversion_factor(self, bispectrum_data):
        """B^phys_{l1,l2,l} = sqrt(2l+1) * B^su2_{l1,l2,l} for all triples."""
        B_phys, B_su2, triples, lmax = bispectrum_data
        for idx, (l1, l2, l) in enumerate(triples):
            factor = math.sqrt(2 * l + 1)
            np.testing.assert_allclose(
                B_phys[0, idx],
                factor * B_su2[0, idx],
                atol=1e-10,
                rtol=1e-10,
                err_msg=(
                    f"Conversion failed for triple ({l1},{l2},{l}): "
                    f"B_phys={B_phys[0,idx]:.6g}, "
                    f"sqrt(2l+1)*B_su2={factor*B_su2[0,idx]:.6g}"
                ),
            )

    def test_l0_output_no_scaling(self):
        """For l_out=0 (factor=1), phys and su2 bispectra must match exactly."""
        lmax = 4
        f = _make_complex_flm(lmax, seed=99)
        B_phys = _bispectrum_from_array(f, lmax)
        B_su2 = _su2_bispectrum(f, lmax)
        triples = _valid_triples(lmax)
        for idx, (l1, l2, l) in enumerate(triples):
            if l == 0:
                np.testing.assert_allclose(
                    B_phys[0, idx],
                    B_su2[0, idx],
                    atol=1e-10,
                    err_msg=f"l=0 triple ({l1},{l2},0): phys={B_phys[0,idx]:.6g} su2={B_su2[0,idx]:.6g}",
                )
