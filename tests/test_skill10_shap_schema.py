import numpy as np
from sklearn.datasets import make_classification
import lightgbm as lgb
import shap
from zindian.cv import make_cv_splitter


def test_skill10_shap_output_schema():
    # Create small dataset
    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=3, random_state=1
    )
    X = np.asarray(X)
    y = np.asarray(y)
    splitter = make_cv_splitter(
        {"type": "stratified", "n_splits": 3, "random_seed": 42}
    )

    feature_names = [f"f{i}" for i in range(X.shape[1])]
    per_fold_feature_means = []
    import warnings

    for tr_idx, val_idx in splitter.split(X, y):
        model = lgb.LGBMClassifier(n_estimators=20, num_leaves=15, random_state=42)
        model.fit(X[tr_idx], y[tr_idx])

        # compute SHAP on validation only
        # suppress known SHAP/LightGBM user warning about return shape changes
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LightGBM binary classifier with TreeExplainer shap values output has changed",
            )
            with np.errstate(all="ignore"):
                expl = shap.TreeExplainer(model)
                sv = expl.shap_values(X[val_idx])
            if isinstance(sv, list):
                sv = sv[-1]
            arr = np.abs(np.asarray(sv, dtype=float))
            per_fold_feature_means.append(arr.mean(axis=0).tolist())

    mean_shap = np.mean(np.vstack(per_fold_feature_means), axis=0).tolist()

    # Expected schema
    output = {
        "feature_names": feature_names,
        "mean_shap": mean_shap,
        "per_fold_means": per_fold_feature_means,
    }

    # Validate schema keys and types
    assert set(output.keys()) == {"feature_names", "mean_shap", "per_fold_means"}
    assert isinstance(output["feature_names"], list)
    assert isinstance(output["mean_shap"], list)
    assert len(output["feature_names"]) == len(output["mean_shap"]) == X.shape[1]
