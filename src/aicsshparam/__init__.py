# -*- coding: utf-8 -*-

"""Top-level package for aics-shparam."""

__author__ = "Matheus Viana"
__email__ = "matheus.viana@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.1.11"

__all__ = [
    "get_module_version",
    "get_shcoeffs_from_mesh",
    "get_shcoeffs_from_vertices_faces",
    "get_invariants",
    "power_spectrum",
    "bispectrum",
    "get_so2_invariants",
    "so2_power_spectrum",
    "so2_cross_power",
    "so2_bispectrum",
]

from .shparam import get_shcoeffs_from_mesh, get_shcoeffs_from_vertices_faces  # noqa: E402,F401
from .shinvariants import (
    get_invariants,
    power_spectrum,
    bispectrum,
    get_so2_invariants,
    so2_power_spectrum,
    so2_cross_power,
    so2_bispectrum,
)


def get_module_version():
    return __version__
