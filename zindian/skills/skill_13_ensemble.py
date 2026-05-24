"""
Compatibility shim: expose `skill_13_ensemble` module name for tests and callers.
Delegates to `skill_13_oracle_fusion` implementation.
"""
from .skill_13_oracle_fusion import run

__all__ = ["run"]
