"""Model evaluation: calibration, metrics, and plots."""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score

from config.settings import EVAL_DIR


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / total * abs(bin_acc - bin_conf)

    return ece


def evaluate_model(model, X, y, name: str) -> dict:
    """Compute all metrics for a model on a dataset."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    y_arr = np.array(y)

    metrics = {
        "accuracy": accuracy_score(y_arr, y_pred),
        "brier_score": brier_score_loss(y_arr, y_prob),
        "log_loss": log_loss(y_arr, y_prob),
        "auc_roc": roc_auc_score(y_arr, y_prob),
        "ece": expected_calibration_error(y_arr, y_prob),
    }

    return metrics


def plot_calibration(y_true, y_probs_dict: dict, save_path: Path = None):
    """Plot calibration curves for multiple models.

    Args:
        y_true: true labels
        y_probs_dict: {model_name: y_prob_array}
        save_path: where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for name, y_prob in y_probs_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
        ax1.plot(prob_pred, prob_true, marker="o", label=name)

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title("Calibration Curve (Reliability Diagram)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Histogram of predicted probabilities
    for name, y_prob in y_probs_dict.items():
        ax2.hist(y_prob, bins=30, alpha=0.5, label=name)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predictions")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved calibration plot to {save_path}")
    plt.close(fig)


def plot_feature_importance(model, feature_names: list, save_path: Path = None):
    """Plot feature importance from a tree model."""
    # Handle calibrated classifiers
    estimator = model
    if hasattr(estimator, "estimator"):
        estimator = estimator.estimator
    if hasattr(estimator, "calibrated_classifiers_"):
        estimator = estimator.calibrated_classifiers_[0].estimator

    if not hasattr(estimator, "feature_importances_"):
        print("Model does not support feature_importances_")
        return

    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    names_sorted = [feature_names[i] for i in indices]
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(names_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved feature importance plot to {save_path}")
    plt.close(fig)


def compare_models(results: dict) -> None:
    """Print comparison table."""
    print("\n" + "=" * 70)
    print(f"{'Model':<30} {'Acc':>7} {'Brier':>7} {'LogL':>7} {'AUC':>7} {'ECE':>7}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<30} {m['accuracy']:>7.4f} {m['brier_score']:>7.4f} "
              f"{m['log_loss']:>7.4f} {m['auc_roc']:>7.4f} {m['ece']:>7.4f}")
    print("=" * 70)
