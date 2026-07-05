import numpy as np
from sklearn.datasets import make_classification
import lightgbm as lgb
import shap
from zindian.cv import make_cv_splitter


def test_shap_computed_per_fold():
    X, y = make_classification(
        n_samples=120, n_features=6, n_informative=3, random_state=0
    )
    X = np.asarray(X)
    y = np.asarray(y)
    n_splits = 3
    splitter = make_cv_splitter(
        {"type": "stratified", "n_splits": n_splits, "random_seed": 42}
    )

    per_fold_means = []
    import warnings

    for tr_idx, val_idx in splitter.split(X, y):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val = X[val_idx]

        # use a very small LightGBM model for speed
        model = lgb.LGBMClassifier(n_estimators=10, num_leaves=15, random_state=42)
        model.fit(X_tr, y_tr)

        # Suppress known SHAP/LightGBM user warning about return shape changes
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LightGBM binary classifier with TreeExplainer shap values output has changed",
            )
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X_val)
        # shap returns list for multioutput; pick final if so
        if isinstance(sv, list):
            sv = sv[-1]
        arr = np.abs(np.asarray(sv, dtype=float))
        assert arr.ndim == 2
        per_fold_means.append(arr.mean(axis=0))

    mean_across_folds = np.mean(np.vstack(per_fold_means), axis=0)
    assert mean_across_folds.shape[0] == X.shape[1]
    assert np.all(mean_across_folds >= 0)
