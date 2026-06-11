import os


def pytest_sessionstart(session):
    """Disable network access for tests by default.

    Set `ZINDIAN_ALLOW_NETWORK=1` in the environment to opt out
    when you intentionally want network-enabled tests.
    """
    if os.environ.get("ZINDIAN_ALLOW_NETWORK"):
        return
    os.environ["ZINDIAN_DISABLE_NETWORK"] = "1"
