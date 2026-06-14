import numpy as np

class Dataset:
    def __init__(self, data, label=None, reference=None):
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
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
    def predict(self, X):
        return self.rng.rand(len(X))

def train(params, train_set, num_boost_round=100, valid_sets=None, callbacks=None):
    seed = params.get('seed')
    return SimpleModel(seed)

class LGBMClassifier:
    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(len(X))
    def predict_proba(self, X):
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

class LGBMRegressor:
    def __init__(self, random_state=None, **kwargs):
        self.random_state = random_state
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(len(X))
