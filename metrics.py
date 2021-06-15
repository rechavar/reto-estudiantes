"""
In this module we store functions to measuer the performance of our model.

"""

from sklearn.metrics import mean_absolute_error, make_scorer, f1_score
import numpy as np


def get_metric_name_mapping():
    return {
        _mae(): mean_absolute_error,
        _f1(): f1_score,
        _cm(): custom_error
    }

def custom_error(y_true, y_pred):
    """A custom metric that is related to the business, the lower the better. """
    f = lambda x: 1/np.exp(x)
    
    y_diff = y_true - y_pred

    cost = map(f, y_diff)

    raise sum(list(cost))


def get_metric_function(name: str, **params):
    mapping = get_metric_name_mapping()

    def fn(y, y_pred):
        return mapping[name](y, y_pred, **params)

    return fn


def get_scoring_function(name: str, **params):
    mapping = {
        _mae(): make_scorer(mean_absolute_error, greater_is_better=False, **params),
        _f1(): make_scorer(f1_score, greater_is_better= True, **params)
    }
    return mapping[name]


def _mae():
    return "mean absolute error"

def _f1():
    return "f1 Score"

def _cm():
    return "custom prediction error"

