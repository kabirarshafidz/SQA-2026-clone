from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix

def compute_metrics(y_true, y_pred, y_scores):
    accuracy = accuracy_score(y_true, y_pred, pos_label=1)
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    pr_auc = average_precision_score(y_true == 1, y_scores, pos_label=1)
    confusion_matrix = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix.tolist()  # convert to list for easier serialization
    }

    return metrics