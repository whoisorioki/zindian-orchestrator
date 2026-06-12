from zindian.zindi_monitor_core import _parse_deadline, _resolve_external_banned


def test_external_data_defaults_to_restricted_when_ambiguous():
    assert (
        _resolve_external_banned("competition rules without external-data language")
        is True
    )


def test_external_data_explicit_permission_overrides_restrictive_default():
    assert (
        _resolve_external_banned("external data is allowed for this challenge") is False
    )


def test_deadline_parser_uses_calendar_month_names():
    assert (
        _parse_deadline("The challenge closes on November 12, 2026")
        == "November 12, 2026"
    )
