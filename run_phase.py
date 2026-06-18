#!/usr/bin/env python3
"""Simple phase runner for World Cup 2026 validation test."""
import sys
import os

# Set competition context
os.environ['ZINDIAN_COMPETITION'] = 'world-cup-2026-goal-prediction-challenge'

from zindian.orchestrator import run_phase

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_phase.py <phase>")
        print("Example: python run_phase.py 1")
        sys.exit(1)
    
    phase = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"EXECUTING PHASE {phase}")
    print(f"{'='*60}\n")
    
    results = run_phase(phase)
    
    print(f"\n{'='*60}")
    print(f"PHASE {phase} RESULTS")
    print(f"{'='*60}")
    for skill, result in results.items():
        status = result.get('status', 'UNKNOWN')
        print(f"\n{skill}: {status}")
        if status == "ERROR":
            print(f"  Error: {result.get('message', 'Unknown error')}")
