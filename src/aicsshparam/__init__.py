# -*- coding: utf-8 -*-

"""Top-level package for aics-shparam."""

__author__ = "Matheus Viana"
__email__ = "matheus.viana@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.1.11"

__all__ = ["get_module_version", "get_invariants", "power_spectrum", "bispectrum"]

from .shinvariants import get_invariants, power_spectrum, bispectrum


def get_module_version():
    return __version__
