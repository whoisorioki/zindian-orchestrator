#!/usr/bin/env python3
"""Inspect the real Zindi API."""

from zindi.user import Zindian
import inspect

print("=" * 80)
print("ZINDIAN API INSPECTION")
print("=" * 80)

print("\n[1] Constructor Signature:")
print(f"  {inspect.signature(Zindian.__init__)}")

print("\n[2] Public Methods & Attributes:")
members = inspect.getmembers(Zindian)
methods = [m for m in members if not m[0].startswith("_")]
for name, obj in methods[:30]:
    if callable(obj):
        try:
            sig = inspect.signature(obj)
            print(f"  {name}{sig}")
        except Exception:
            print(f"  {name} (callable, signature unavailable)")
    else:
        print(f"  {name} = {type(obj).__name__}")

print("\n[3] Docstring (if available):")
if Zindian.__doc__:
    print(Zindian.__doc__[:500])
else:
    print("  (No docstring)")

print("\n[4] __init__ Docstring:")
if Zindian.__init__.__doc__:
    print(Zindian.__init__.__doc__[:800])
else:
    print("  (No docstring)")
