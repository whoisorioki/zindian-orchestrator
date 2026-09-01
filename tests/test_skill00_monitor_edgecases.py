from zindian.zindi_monitor_core import _resolve_external_banned, _parse_deadline


def test_resolve_external_banned_ambiguous_returns_none():
    text = "This challenge has no explicit external data statement"
    # Unresolved must be None, not a silent True — operator is the decider.
    assert _resolve_external_banned(text) is None


def test_resolve_external_banned_allows():
    text = "external data is allowed for this challenge"
    assert _resolve_external_banned(text) is False


def test_resolve_external_banned_encouragement_permits():
    text = "You are welcome to download additional climate features"
    assert _resolve_external_banned(text) is False


def test_resolve_external_banned_explicit_ban_wins():
    text = "no external data — you may use only the datasets provided"
    assert _resolve_external_banned(text) is True


def test_resolve_external_banned_conflict_returns_none():
    """Generic ban clause + competition-specific permission => a genuine rule
    conflict that must be left unresolved for the operator, not silently banned."""
    text = (
        "you may use only the datasets provided for this challenge. "
        "Participants are encouraged to use publicly available climate datasets."
    )
    assert _resolve_external_banned(text) is None


def test_parse_deadline_basic():
    s = "The competition closes on May 10, 2026 at 23:59"
    dl = _parse_deadline(s)
    assert dl is not None
    assert "May" in dl and "2026" in dl
