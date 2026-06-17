# Package Isolation Policy

**Version:** 1.0  
**Last Updated:** 2026-06-17  
**Authority:** Binding for all competition ingestions

---

## 1. Purpose

Prevent local directory shadowing of system packages that causes import resolution failures and silent behavioral changes in production environments.

---

## 2. Prohibited Patterns

### 2.1 Local Mock Stubs

**FORBIDDEN:**
```
workspace/
├── duckdb/          ← Shadows site-packages duckdb
│   └── __init__.py
├── google/          ← Shadows site-packages google
│   └── genai/
└── zindi/           ← Shadows site-packages zindi
    └── user.py
```

**Rationale:** Python's import resolution prioritizes `sys.path[0]` (current working directory) over `site-packages`. Local directories named identically to installed packages will shadow the real implementations.

### 2.2 Naming Boundaries

All local mock/stub directories MUST use suffixed naming:
- `duckdb_local/` or `duckdb_stub/`
- `google_mock/` or `google_local/`
- `zindi_stub/` or `zindi_local/`

---

## 3. Verification Commands

### 3.1 Pre-Flight Check

```bash
python3 -c "import duckdb; print(duckdb.__file__)"
python3 -c "import google.genai; print(google.genai.__file__)"
python3 -c "import zindi; print(zindi.__file__)"
```

**Expected Output:** All paths MUST resolve to `/opt/conda/lib/python3.X/site-packages/...`

**Failure Indicator:** Paths resolving to workspace directories indicate shadowing.

### 3.2 Automated Enforcement

Add to `scripts/preflight_enforce.py`:

```python
def check_package_shadowing():
    """Verify no local directories shadow installed packages."""
    workspace_root = Path.cwd()
    forbidden_names = {"duckdb", "google", "zindi", "pandas", "numpy"}
    
    for name in forbidden_names:
        local_path = workspace_root / name
        if local_path.exists() and local_path.is_dir():
            if not name.endswith(("_local", "_stub", "_mock")):
                raise RuntimeError(
                    f"Package shadowing detected: {local_path} shadows site-packages/{name}"
                )
```

---

## 4. Remediation Protocol

When shadowing is detected:

1. **Rename** the local directory with `_DISABLED` suffix
2. **Verify** imports resolve to site-packages
3. **Update** all references in code/docs
4. **Commit** with message: `fix: eliminate {package} shadowing`

---

## 5. Enforcement

- **Preflight:** MUST pass before any skill execution
- **CI/CD:** Block merges if shadowing detected
- **Code Review:** Reject PRs introducing forbidden patterns

---

## 6. References

- Session Log: `docs/session_logs/PACKAGE_SHADOWING_AUDIT.md`
- Python Import System: https://docs.python.org/3/reference/import.html
