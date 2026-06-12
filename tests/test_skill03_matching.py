from zindian.skills.skill_03_legality import check_planned_features


def test_planned_feature_transform_not_blocked_on_substring():
    policy = {
        "allowed_data_sources": ["competition_provided_only"],
        "banned_transformations": ["dem"],
        "coordinate_features_permitted": True,
        "external_data_permitted": False,
    }

    planned = [
        {"name": "feat1", "transforms": ["academic_demand"], "uses_lat_lon": False}
    ]
    results = check_planned_features(policy, planned)
    assert results[0]["status"] != "BLOCK"
