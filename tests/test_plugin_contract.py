"""Test FeatureExtractor ABC contract."""

import pytest
from plugins.base_extractor import FeatureExtractor
from plugins.geoai_extractor import Extractor as GeoAIExtractor


def test_base_extractor_is_abc():
    """FeatureExtractor should be abstract."""
    with pytest.raises(TypeError):
        FeatureExtractor()  # type: ignore[abstract]


def test_geoai_inherits_from_abc():
    """GeoAI extractor should inherit from FeatureExtractor."""
    extractor = GeoAIExtractor()
    assert isinstance(extractor, FeatureExtractor)
    assert hasattr(extractor, "fetch")
    assert hasattr(extractor, "extract")


def test_extractor_has_required_methods():
    """Extractor must implement fetch and extract."""
    extractor = GeoAIExtractor()

    # Check methods exist
    assert callable(extractor.fetch)
    assert callable(extractor.extract)

    # Check signatures
    import inspect

    fetch_sig = inspect.signature(extractor.fetch)
    assert "paths" in fetch_sig.parameters
    assert "config" in fetch_sig.parameters

    extract_sig = inspect.signature(extractor.extract)
    assert "paths" in extract_sig.parameters
    assert "config" in extract_sig.parameters
