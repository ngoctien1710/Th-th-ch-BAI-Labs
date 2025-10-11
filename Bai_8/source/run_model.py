from EDA import EDA_train, EDA_test
from report_score import report_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import lightgbm as lg
from typing import Any
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from no_EDA import no_EDA_test, no_EDA_train
from sklearn.base import clone
import time


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Thời gian chạy: {end - start:.3f} giây\n")
    return wrapper


@timed
def model_with_VAR(model: Any, train_data, test_data) -> None:
    X_train, y_train, scaler = no_EDA_train(train_data)
    X_test, y_test = no_EDA_test(test_data, X_train.columns, scaler)

    var = VarianceThreshold(threshold=0.011)
    X_train = var.fit_transform(X_train)
    X_test = var.transform(X_test)

    model.fit(X_train, y_train)
    report_score(model, X_test, y_test, 'VarianceThreshold')


@timed
def model_with_PCA(model: Any, train_data, test_data) -> None:
    X_train, y_train, scaler = no_EDA_train(train_data)
    X_test, y_test = no_EDA_test(test_data, X_train.columns, scaler)

    pca = PCA(n_components=4)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    model.fit(X_train, y_train)
    report_score(model, X_test, y_test, 'PCA')


@timed
def model_with_PRO(model: Any, train_data, test_data) -> None:
    X_train, y_train, scaler = EDA_train(train_data)
    X_test, y_test = EDA_test(test_data, X_train.columns, scaler)

    model.fit(X_train, y_train)
    report_score(model, X_test, y_test, 'Preprocessing')


def run(models: list[Any], train_data, test_data) -> None:
    for model in models:
        print(f"\n================ {model.__class__.__name__} =================")

        print("→ Using Full Preprocessing:")
        model_with_PRO(clone(model), train_data, test_data)
        
        print("→ Using PCA:")
        model_with_PCA(clone(model), train_data, test_data)

        print("→ Using Variance Threshold:")
        model_with_VAR(clone(model), train_data, test_data)
