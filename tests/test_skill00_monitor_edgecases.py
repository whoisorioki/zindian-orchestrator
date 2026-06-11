from zindian.skills.skill_00_zindi_monitor import _resolve_external_banned, _parse_deadline


def test_resolve_external_banned_defaults_true():
    text = "This challenge has no explicit external data statement"
    assert _resolve_external_banned(text) is True


def test_resolve_external_banned_allows():
    text = "external data is allowed for this challenge"
    assert _resolve_external_banned(text) is False


def test_parse_deadline_basic():
    s = "The competition closes on May 10, 2026 at 23:59"
    dl = _parse_deadline(s)
    assert dl is not None
    assert "May" in dl and "2026" in dl
