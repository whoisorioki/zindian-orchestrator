import numpy as np

class TreeExplainer:
    """Minimal stub of shap.TreeExplainer used for CI tests.
    Returns zero SHAP values for any model.
    """

    def __init__(self, model, data=None, model_output="logit"):
        self.model = model
        self.model_output = model_output

    def shap_values(self, X, check_additivity=False):
        """Return zero SHAP values matching expected dimensions.
        For binary classification, LightGBM models return a list with two arrays.
        This stub returns a list of two zero arrays.
        """
        n_samples, n_features = X.shape
        return [np.zeros((n_samples, n_features)), np.zeros((n_samples, n_features))]
