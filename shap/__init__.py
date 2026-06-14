import numpy as np
from lightgbm import LGBMClassifier

class TreeExplainer:
    """Hardened stub of shap.TreeExplainer used for CI tests.
    Validates feature dimensions against the fitted model.
    """

    def __init__(self, model, data=None, model_output="logit"):
        self.model = model
        self.model_output = model_output

    def shap_values(self, X, check_additivity=False):
        """Return zero SHAP values matching expected dimensions.
        Validates feature dimensions against the fitted model and returns
        classifier-specific list of arrays or regressor-specific single array.
        """
        X_arr = np.asarray(X)
        n_samples, n_features = X_arr.shape
        
        # Check that X feature count matches model's fitted feature count
        if hasattr(self.model, "n_features_in_") and self.model.n_features_in_ is not None:
            if n_features != self.model.n_features_in_:
                raise ValueError(
                    f"Dimension mismatch: model expects {self.model.n_features_in_} features, but got {n_features}"
                )
        elif hasattr(self.model, "n_features") and self.model.n_features is not None:
            if n_features != self.model.n_features:
                raise ValueError(
                    f"Dimension mismatch: model expects {self.model.n_features} features, but got {n_features}"
                )

        if isinstance(self.model, LGBMClassifier):
            # For classification, return list of two arrays
            return [np.zeros((n_samples, n_features)), np.zeros((n_samples, n_features))]
        
        # For regression, return a single array
        return np.zeros((n_samples, n_features))
