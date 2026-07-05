"""Test R5 carbon tracking telemetry."""

from zindian.carbon_tracker import estimate_carbon


def test_carbon_estimate_fallback():
    """Test ML CO2 formula fallback when CodeCarbon unavailable."""
    config = {
        "infrastructure": {
            "hardware_type": "cpu",
            "region": "us-east-1",
            "tdp_watts": 15.0,
            "pue": 1.0,
            "carbon_intensity_gco2_per_kwh": 494.0,
        }
    }

    result = estimate_carbon(duration_sec=10.0, peak_memory_mb=100.0, config=config)

    assert "carbon_kg_estimate" in result
    assert "tracker_method" in result
    assert "hardware_type" in result
    assert "region" in result

    # Verify formula: (15 * 1.0 * 10) / 3_600_000 * 494 / 1000
    expected = (15.0 * 1.0 * 10.0) / 3_600_000 * 494.0 / 1000
    assert abs(result["carbon_kg_estimate"] - expected) < 1e-9
    assert result["tracker_method"] in ["codecarbon", "mlco2_formula"]
    assert result["hardware_type"] == "cpu"
    assert result["region"] == "us-east-1"


def test_carbon_estimate_defaults():
    """Test defaults when infrastructure block missing."""
    config = {}

    result = estimate_carbon(duration_sec=5.0, peak_memory_mb=50.0, config=config)

    assert result["carbon_kg_estimate"] is not None
    assert result["tracker_method"] in [
        "codecarbon",
        "mlco2_formula",
        "not_instrumented",
    ]
    # Should use defaults: tdp=15, pue=1.0, carbon_intensity=494
    expected = (15.0 * 1.0 * 5.0) / 3_600_000 * 494.0 / 1000
    assert abs(result["carbon_kg_estimate"] - expected) < 1e-9


def test_telemetry_schema():
    """Verify telemetry includes all R5 fields."""
    from zindian.orchestrator import run_skill

    # Mock a simple skill run
    result = run_skill("skill_01")

    assert "telemetry" in result
    telemetry = result["telemetry"]

    # Standard fields
    assert "duration_sec" in telemetry
    assert "peak_memory_mb" in telemetry

    # R5 fields
    assert (
        "carbon_kg_estimate" in telemetry
        or telemetry.get("tracker_method") == "not_instrumented"
    )
    assert "tracker_method" in telemetry
    assert "hardware_type" in telemetry
    assert "region" in telemetry
