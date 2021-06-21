"""
Here we code what our model is. It may include all of feature engineering.
"""
import typing as t

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.cluster import KMeans


EstimatorConfig = t.List[t.Dict[str, t.Any]]


def build_estimator(config: EstimatorConfig):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for step in config:
        name = step["name"]
        hparams = step.get("params", {})
        estimator = estimator_mapping[name](**hparams)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "random-forest-classifier": RandomForestClassifier,
        "KMeans": KMeans,
        "logistic_regression": LogisticRegression,
        "base_line": BaseLine
    }


class BaseLine(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.weights = []
        self.y_pred = []
        self.score_true = []

    def fit(self, X, y):
        total = len(X.index)
        columns_data = X.columns
        for i in columns_data:
            if i == 'age' or i == 'trtbps' or i == 'chol' or i == 'thalachh' or i =='oldpeak' or i =='fbs' or i == 'exng':
                self.weights.append(1)
            else:
                w_temp = X[i].value_counts()
                w_temp = (w_temp/ total)
                self.weights.append(w_temp)

    def predict(self, X):
        for column, weight in zip(X.columns, self.weights):
            if type(weight) != int:
                aux = []
                for value in X[column]:
                    aux.append((value * weight[value]))

                self.score_true.append(aux)
            else:
                self.score_true.append(list(X[column].values * weight))
        

        score_true = np.asarray(self.score_true).sum(axis=0)

        threshold = (max(score_true) - min(score_true))/2

        # pred
        for x in score_true:
            if x > threshold:
                self.y_pred.append(1)
            else:
                self.y_pred.append(0)

        return self.y_pred
