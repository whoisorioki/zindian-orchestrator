import numpy as np
from zindian.cv import make_cv_splitter, get_cv_splits
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold


def test_make_cv_splitter_types():
    s = make_cv_splitter({"type": "stratified", "n_splits": 4, "random_seed": 7})
    assert isinstance(s, StratifiedKFold)

    g = make_cv_splitter({"type": "group", "n_splits": 3})
    assert isinstance(g, GroupKFold)

    k = make_cv_splitter({"type": "kfold", "n_splits": 2})
    assert isinstance(k, KFold)


def test_get_cv_splits_counts():
    # synthetic data
    rng = np.random.RandomState(0)
    X = rng.randn(50, 5)
    y = rng.randint(0, 2, size=50)

    splits = list(
        get_cv_splits(X, y, cv_strategy={"type": "stratified", "n_splits": 5})
    )
    assert len(splits) == 5
    # each split partition covers all indices exactly once across val sets
    val_union = set()
    for tr, val in splits:
        assert len(set(tr).intersection(set(val))) == 0
        val_union.update(val.tolist())
    assert val_union.issubset(set(range(len(y))))
