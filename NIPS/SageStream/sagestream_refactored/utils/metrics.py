"""
Utility functions and metrics computation for SageStream.
"""

import warnings
from typing import Dict, List, Any, Optional
import random
import gc
import os

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, cohen_kappa_score, recall_score,
    precision_score, matthews_corrcoef, average_precision_score, jaccard_score
)
from sklearn.exceptions import UndefinedMetricWarning

# Filter sklearn warnings about undefined metrics
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def set_all_seeds(seed: int):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def compute_metrics(
    y_true: List,
    y_pred: List,
    y_prob: np.ndarray = None,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        num_classes: Number of classes

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Suppress sklearn warnings during metric computation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Optional metrics with safe computation
        try:
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
            metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
        except Exception:
            metrics['matthews_corrcoef'] = 0.0
            metrics['jaccard_macro'] = 0.0

        # Probability-based metrics
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                if num_classes == 2:
                    prob_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                    metrics['roc_auc'] = roc_auc_score(y_true, prob_pos)
                    metrics['average_precision'] = average_precision_score(y_true, prob_pos)
                else:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    metrics['average_precision_macro'] = 0.0
            except Exception:
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
            metrics['average_precision'] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for each line
    """
    print(f"{prefix}Accuracy:          {metrics.get('accuracy', 0.0):.4f}")
    print(f"{prefix}Balanced Accuracy: {metrics.get('balanced_accuracy', 0.0):.4f}")
    print(f"{prefix}F1 Macro:          {metrics.get('f1_macro', 0.0):.4f}")
    print(f"{prefix}Precision Macro:   {metrics.get('precision_macro', 0.0):.4f}")
    print(f"{prefix}Recall Macro:      {metrics.get('recall_macro', 0.0):.4f}")


def aggregate_fold_results(
    fold_results: List[Dict[str, Any]],
    metric_keys: List[str] = None
) -> Dict[str, Any]:
    """
    Aggregate results from multiple folds.

    Args:
        fold_results: List of results from each fold
        metric_keys: Keys to aggregate (default: common metrics)

    Returns:
        Dictionary with mean and std of metrics
    """
    if metric_keys is None:
        metric_keys = ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']

    aggregated = {}

    for key in metric_keys:
        values = []
        for result in fold_results:
            if 'test_metrics' in result and key in result['test_metrics']:
                values.append(result['test_metrics'][key])
            elif key in result:
                values.append(result[key])

        if values:
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
            aggregated[f'{key}_values'] = values

    return aggregated
