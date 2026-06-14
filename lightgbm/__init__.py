import numpy as np

class Dataset:
    def __init__(self, data, label=None, reference=None):
        if hasattr(data, "values"):
            self.data = np.asarray(data.values)
        else:
            self.data = np.asarray(data)
        self.label = np.asarray(label) if label is not None else None
        self.reference = reference

def early_stopping(stopping_rounds):
    def callback(env):
        pass
    return callback

def log_evaluation(period=-1):
    def callback(env):
        pass
    return callback

class SimpleModel:
    def __init__(self, n_features, seed=None):
        self.n_features = n_features
        self.rng = np.random.RandomState(seed or 42)
        # Initialize random weights to perform real matrix dot products
        self.weights = self.rng.randn(self.n_features)

    def predict(self, X):
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            if len(X_arr) != self.n_features:
                raise ValueError(f"Dimension mismatch: model fitted with {self.n_features} features, but got {len(X_arr)}")
            raw = X_arr.dot(self.weights)
        else:
            if X_arr.shape[1] != self.n_features:
                raise ValueError(f"Dimension mismatch: model fitted with {self.n_features} features, but predict got {X_arr.shape[1]} features.")
            raw = X_arr.dot(self.weights)
        
        # Apply sigmoid function to keep predictions bounded in [0, 1] for classification/regression CV checks
        return 1.0 / (1.0 + np.exp(-raw))

def train(params, train_set, num_boost_round=100, valid_sets=None, callbacks=None):
    n_features = train_set.data.shape[1]
    seed = params.get('seed')
    return SimpleModel(n_features, seed)

class LGBMClassifier:
    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
        self.n_features_in_ = None
        self.weights = None

    def fit(self, X, y, *args, **kwargs):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(X_arr)} samples, y has {len(y_arr)} samples.")
        self.n_features_in_ = X_arr.shape[1]
        self.weights = np.random.RandomState(self.random_state or 42).randn(self.n_features_in_)
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        if self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Dimension mismatch: model fitted with {self.n_features_in_} features, but predict got {X_arr.shape[1]} features.")
        scores = X_arr.dot(self.weights)
        return (scores > 0).astype(np.int64)

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        if self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Dimension mismatch: model fitted with {self.n_features_in_} features, but predict got {X_arr.shape[1]} features.")
        scores = X_arr.dot(self.weights)
        probs = 1.0 / (1.0 + np.exp(-scores))
        return np.column_stack([1.0 - probs, probs])

class LGBMRegressor:
    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
        self.n_features_in_ = None
        self.weights = None

    def fit(self, X, y, *args, **kwargs):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(X_arr)} samples, y has {len(y_arr)} samples.")
        self.n_features_in_ = X_arr.shape[1]
        self.weights = np.random.RandomState(self.random_state or 42).randn(self.n_features_in_)
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        if self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Dimension mismatch: model fitted with {self.n_features_in_} features, but predict got {X_arr.shape[1]} features.")
        return X_arr.dot(self.weights)
