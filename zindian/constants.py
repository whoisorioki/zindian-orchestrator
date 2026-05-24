"""
Central repository constants to avoid cross-skill imports.
Place shared, competition-agnostic constants here (TC_VARIABLES, bands, regions).
"""
from __future__ import annotations

TC_VARIABLES = [
    "aet", "def", "pdsi", "pet", "ppt",
    "q", "soil", "srad", "swe",
    "tmax", "tmin", "vap", "vpd",
]

TC_STATS = ["mean", "std", "min", "max"]
TC_BAND_NAMES = [f"{v}_{s}" for v in TC_VARIABLES for s in TC_STATS]

# Bbox — SE Australia (confirmed 100% coverage of training data)
MIN_LON, MAX_LON = 139.94, 151.48
MIN_LAT, MAX_LAT = -39.74, -30.92
TIME_SLICE = ("2011-01-01", "2021-12-01")
