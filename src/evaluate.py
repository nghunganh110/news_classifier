import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict


def plot_confusion_matrix(y_true, y_pred, labels: List[str], title: str, output_path: str):
    """Create and save a confusion matrix plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")


def print_classification_report(y_true, y_pred, labels: List[str]):
    """Print a formatted classification report."""
    print(classification_report(y_true, y_pred, target_names=labels))


def compare_models(results_dict: Dict[str, float]):
    """Print a comparison table of model accuracies."""
    print("\n=== Model Comparison ===")
    print(f"{'Model':<30} {'Accuracy':>10}")
    print("-" * 42)
    for model_name, accuracy in sorted(results_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:<30} {accuracy:>10.4f}")
    print()


class ModelEvaluator:
    """Wraps evaluation utilities for a trained model."""

    def __init__(self, labels: List[str]):
        self.labels = labels

    def evaluate(self, y_true, y_pred, model_name: str = "Model", output_dir: str = "outputs"):
        """Full evaluation: report + confusion matrix."""
        print(f"\n=== {model_name} Evaluation ===")
        print_classification_report(y_true, y_pred, self.labels)
        output_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, self.labels, f"{model_name} Confusion Matrix", output_path)
        acc = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"Accuracy: {acc:.4f}")
        return acc
