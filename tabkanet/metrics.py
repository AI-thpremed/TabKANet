import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np

def f1_score_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average='macro')

def root_mean_squared_logarithmic_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))


def auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Area Under the Curve (AUC) score.

    Parameters:
    - y_true (np.ndarray): True binary labels.
    - y_pred (np.ndarray): Predicted probabilities.

    Returns:
    - float: AUC score.
    """
    return roc_auc_score(y_true, y_pred)