"""
In this module we store prepare the dataset for machine learning experiments.
"""

import typing as t
import typing_extensions as te

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df_orginal = reader()
    df = clean_dataset(df_orginal)
    y = df_orginal["y"]
    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaning_fn = _chain(
        [
             _fix_data_frame_cat,
             _add_new_features,
             _fix_data_frame_con

        ]
    )
    df = cleaning_fn(df)
    return df


def _chain(functions: t.List[t.Callable[[pd.DataFrame], pd.DataFrame]]):
    def helper(df):
        for fn in functions:
            df = fn(df)
        return df

    return helper

def _add_new_features(df):
    df['new_feature_1'] = np.where(df['thalachh'] > 130, 0, np.where(df['oldpeak'] == 0, 1, 0))
    df['new_feature_2'] = np.where(df['thalachh'] > 130, 0, np.where(df['exng_1'] == 0, 1, 0))
    return df


def _fix_data_frame_cat(df):
    to_get_dummies_cols = get_categorical_column_names()
    df_cat_dummies = pd.get_dummies(df[to_get_dummies_cols], columns= to_get_dummies_cols)
    df_cat_dummies

    continius_cols = get_numeric_column_names()
    
    return pd.concat([df[continius_cols], df_cat_dummies], axis = 1)

def _fix_data_frame_con(df):

    sc = StandardScaler()
    cols_to_scale = ['thalachh', 'age', 'chol', 'trtbps']
    df[cols_to_scale] = sc.fit_transform(df[cols_to_scale])

    return df


def _fix_unhandled_nulls(df):
    df.dropna(inplace=True)
    return df


def get_categorical_column_names() -> t.List[str]:
     return (
         "sex,cp,fbs,restecg,exng,slp,caa,thall"
        
     ).split(",")


def get_binary_column_names() -> t.List[str]:
    return ("sex,cp,restecg,slp,thall,thall,fbs,exng").split(",")


def get_numeric_column_names() -> t.List[str]:
    return (
        "age,trtbps,chol,thalachh,oldpeak"
    ).split(",")


def get_column_names() -> t.List[str]:
    return (
        "age,trtbps,chol,thalachh,oldpeak,sex,cp,restecg,slp,thall,thall,fbs,exng"
    ).split(",")


def get_categorical_variables_values_mapping() -> t.Dict[str, t.Sequence[str]]:
    return {
        "cp": ("0","1","2","3"),
        "restecg": ("0","1","2"),
        "slp": ("0","1","2"),
        "caa": ("0", "1", "2", "3"),
        "thall": ("0", "1","2","3"),
         }
