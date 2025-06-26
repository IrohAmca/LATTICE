from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)


def compute_metrics(true_labels, pred_labels, metrics_config=None):
    default_metrics = ["accuracy", "f1", "precision", "recall"]
    metrics_to_calculate = default_metrics

    if metrics_config and "calculate" in metrics_config:
        metrics_to_calculate = metrics_config["calculate"]

    results = {}

    if any(metric in metrics_to_calculate for metric in ["f1", "precision", "recall"]):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted", zero_division=0
        )

        if "precision" in metrics_to_calculate:
            results["precision"] = precision

        if "recall" in metrics_to_calculate:
            results["recall"] = recall

        if "f1" in metrics_to_calculate:
            results["f1"] = f1

    if "accuracy" in metrics_to_calculate:
        results["accuracy"] = accuracy_score(true_labels, pred_labels)

    if "confusion_matrix" in metrics_to_calculate:
        cm = confusion_matrix(true_labels, pred_labels)
        results["confusion_matrix"] = cm

        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Tahmin Edilen")
            plt.ylabel("Gerçek")
            plt.title("Confusion Matrix")

            output_dir = os.path.join("../outputs", "metrics")
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
            plt.close()

        except Exception as e:
            logger.warning(f"Confusion matrix kaydedilemedi: {e}")

    if metrics_config and metrics_config.get("class_metrics", False):
        try:
            prec_class, rec_class, f1_class, support = precision_recall_fscore_support(
                true_labels, pred_labels, average=None, zero_division=0
            )

            unique_labels = np.unique(np.concatenate([true_labels, pred_labels]))

            class_metrics = {}
            for i, label in enumerate(unique_labels):
                class_metrics[str(label)] = {
                    "precision": prec_class[i] if i < len(prec_class) else 0,
                    "recall": rec_class[i] if i < len(rec_class) else 0,
                    "f1": f1_class[i] if i < len(f1_class) else 0,
                    "support": int(support[i]) if i < len(support) else 0,
                }

            results["class_metrics"] = class_metrics

        except Exception as e:
            logger.warning(f"Sınıf metrikleri hesaplanamadı: {e}")

    return results
