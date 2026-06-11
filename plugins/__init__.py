"""Plugins package for dataset-specific feature extraction implementations.
Plugins should expose `fetch(paths, config, allow_network=True)` and
`extract(paths, tiff_path, config)` functions.
"""

__all__ = ["terraclimate_extractor"]
