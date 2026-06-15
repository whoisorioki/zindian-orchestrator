"""
Central repository constants to avoid cross-skill imports.
Place shared, competition-agnostic constants here (TC_VARIABLES, bands, regions).
"""

from __future__ import annotations

TC_VARIABLES = [
    "aet",
    "def",
    "pdsi",
    "pet",
    "ppt",
    "q",
    "soil",
    "srad",
    "swe",
    "tmax",
    "tmin",
    "vap",
    "vpd",
]

TC_STATS = ["mean", "std", "min", "max"]
TC_BAND_NAMES = [f"{v}_{s}" for v in TC_VARIABLES for s in TC_STATS]


