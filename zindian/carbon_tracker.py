"""
R5 Carbon Tracking — v2.3 Feature

Estimates carbon footprint for skill execution using:
- Primary: CodeCarbon (optional dependency)
- Fallback: ML CO2 Impact formula
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def estimate_carbon(
    duration_sec: float,
    peak_memory_mb: float,
    config: dict
) -> Dict[str, Any]:
    """
    Estimate carbon footprint for a skill run.
    
    Args:
        duration_sec: Wall-clock execution time
        peak_memory_mb: Peak memory usage
        config: challenge_config.json with infrastructure block
    
    Returns:
        {
            "carbon_kg_estimate": float,
            "tracker_method": "codecarbon" | "mlco2_formula" | "not_instrumented",
            "hardware_type": "cpu" | "gpu" | "tpu",
            "region": str
        }
    """
    infra = config.get("infrastructure", {})
    
    # Try CodeCarbon first
    try:
        from codecarbon import EmissionsTracker
        tracker = EmissionsTracker(measure_power_secs=duration_sec, save_to_file=False)
        tracker.start()
        # Simulate work duration
        import time
        time.sleep(min(duration_sec, 0.1))  # Cap at 0.1s for estimation
        emissions = tracker.stop()
        
        return {
            "carbon_kg_estimate": float(emissions),
            "tracker_method": "codecarbon",
            "hardware_type": infra.get("hardware_type", "cpu"),
            "region": infra.get("region", "unknown")
        }
    except (ImportError, Exception) as e:
        logger.debug(f"CodeCarbon unavailable: {e}, using ML CO2 formula")
    
    # Fallback: ML CO2 Impact formula
    tdp_watts = float(infra.get("tdp_watts", 15.0))
    pue = float(infra.get("pue", 1.0))
    carbon_intensity = float(infra.get("carbon_intensity_gco2_per_kwh", 494.0))
    
    energy_kwh = (tdp_watts * pue * duration_sec) / 3_600_000
    carbon_kg = (energy_kwh * carbon_intensity) / 1000
    
    return {
        "carbon_kg_estimate": float(carbon_kg),
        "tracker_method": "mlco2_formula",
        "hardware_type": infra.get("hardware_type", "cpu"),
        "region": infra.get("region", "eu-central-1")
    }
