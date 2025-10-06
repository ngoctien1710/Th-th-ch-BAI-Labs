import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pandas.api.types import is_numeric_dtype
from sklearn.cluster import DBSCAN

def kmean_inertia(data: pd.DataFrame, range_cluster: list[int]) -> list:
    inertia = []
    for k in range_cluster:
        kmean = KMeans(n_clusters = k, random_state = 42)
        kmean.fit(data)
        inertia.append(kmean.inertia_)
    return inertia

def kmean_silhouette(data: pd.DataFrame, range_cluster: list[int]) -> list:
    silhouette = []
    for k in range_cluster:
        kmean = KMeans(n_clusters = k, random_state = 42)
        kmean.fit(data)
        silhouette.append(silhouette_score(data, kmean.labels_))
    return silhouette

def outliner(X: pd.DataFrame) -> pd.DataFrame:
    outl = pd.DataFrame()
    for col in X.columns:
        if is_numeric_dtype(X[col]):
            IQR = X[col].quantile(0.75) - X[col].quantile(0.25)
            outl[col] = (X[col] > max(X[col]) + 1.5*IQR) | (X[col] < min(X[col]) - 1.5*IQR )
    print(outl.sum())

def dbscan_silhouette(data: pd.DataFrame, range_cluster: list[float]) -> list:
    silhouette = []
    for eps in range_cluster:
        dbscan = DBSCAN(eps=eps, min_samples = 5).fit_predict(data)
        labels = dbscan[dbscan != -1]
        if len(set(labels)) > 1:
            silhouette.append(silhouette_score(data[dbscan != -1], labels))
        else:
            silhouette.append(-1) 
    return silhouette