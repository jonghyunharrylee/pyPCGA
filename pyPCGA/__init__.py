"""
The pyPCGA package enables you to run matrix-free geostatistical inversion
"""
from __about__ import __version__, __name__, __author__
from .pcga import PCGA
from . import covariance


__all__ = [
    "__version__",
    "__name__",
    "__author__",
    "PCGA",
    "covariance"
    ]
