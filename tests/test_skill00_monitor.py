from zindian.zindi_monitor_core import _parse_deadline, _resolve_external_banned


def test_external_data_ambiguous_returns_none_unresolved():
    # An ambiguous page must NOT silently default to restricted/banned — the
    # caller (monitor / operator) is the decider.
    assert (
        _resolve_external_banned("competition rules without external-data language")
        is None
    )


def test_external_data_explicit_permission_overrides_restrictive_default():
    assert (
        _resolve_external_banned("external data is allowed for this challenge") is False
    )


def test_external_data_encouragement_permits():
    assert (
        _resolve_external_banned(
            "participants are encouraged to use publicly available climate datasets"
        )
        is False
    )


def test_deadline_parser_uses_calendar_month_names():
    assert (
        _parse_deadline("The challenge closes on November 12, 2026")
        == "November 12, 2026"
    )
