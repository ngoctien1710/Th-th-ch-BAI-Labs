from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any

def report_score(model: Any, X: pd.DataFrame, y: pd.Series, pp: str) -> None:
    y_pred = model.predict(X)
    print(f'Precision score: {precision_score(y, y_pred):.2f}')
    print(f'Recall score:    {recall_score(y, y_pred):.2f}')
    print(f'Accuracy score:  {accuracy_score(y, y_pred):.2f}')
    print(f'F1 score:        {f1_score(y, y_pred):.2f}')

    # ROC AUC (nếu mô hình hỗ trợ xác suất)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        print(f'AUC score:       {roc_auc_score(y, y_prob):.2f}')

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model.__class__.__name__} - {pp}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

