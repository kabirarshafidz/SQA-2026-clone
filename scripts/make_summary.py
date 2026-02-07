import os
import json
import pandas as pd


cols = [
    "run_id",
    "sweep_key",
    "sweep_value",
    "model_name",
    "seed",
    "n_samples",
    "anomaly_ratio",
    "label_noise",
    "pr_auc",
    "recall_anomaly",
    "precision_anomaly",
    "f1_anomaly",
    "accuracy",
    "tn",
    "fp",
    "fn",
    "tp",
]

def make_summary(experiments_root="experiments", summary_path="experiments/summary.csv"):
    # iterate over sweep_key directories
        # iterate over sweep_value directories
            # iterate over seed directories
                # iterate over model_name directories

    for sweep_key in os.listdir(experiments_root):
        sweep_key_path = os.path.join(experiments_root, sweep_key)
        if not os.path.isdir(sweep_key_path):
            continue

        for sweep_value in os.listdir(sweep_key_path):
            sweep_value_path = os.path.join(sweep_key_path, sweep_value)
            if not os.path.isdir(sweep_value_path):
                continue

            for seed_dir in os.listdir(sweep_value_path):
                seed_path = os.path.join(sweep_value_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue

                for model_name in os.listdir(seed_path):
                    model_path = os.path.join(seed_path, model_name)
                    if not os.path.isdir(model_path):
                        continue

                    # load json files
                    metrics_path = os.path.join(model_path, "metrics.json")
                    config_path = os.path.join(model_path, "config.json")

                    if not os.path.isfile(metrics_path) or not os.path.isfile(config_path):
                        continue

                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    with open(config_path, "r") as f:
                        config = json.load(f)

                    cm = metrics.get("confusion_matrix", [[0,0],[0,0]])
                    tn, fp = cm[0]
                    fn, tp = cm[1]

                    row = {
                        "run_id": f"{sweep_key}_{sweep_value}_{seed_dir}_{model_name}",
                        "sweep_key": sweep_key,
                        "sweep_value": sweep_value,
                        "model_name": model_name,
                        "seed": int(seed_dir.split("_")[1]),
                        "n_samples": config.get("n_samples", ""),
                        "anomaly_ratio": config.get("anomaly_ratio", ""),
                        "label_noise": config.get("label_noise", ""),
                        "pr_auc": metrics.get("pr_auc", ""),
                        "recall_anomaly": metrics.get("recall", ""),
                        "precision_anomaly": metrics.get("precision", ""),
                        "f1_anomaly": metrics.get("f1_score", ""),
                        "accuracy": metrics.get("accuracy", ""),
                        "tn": tn,
                        "fp": fp,
                        "fn": fn,
                        "tp": tp,
                    }

                    df_row = pd.DataFrame([row])
                    if not os.path.isfile(summary_path):
                        df_row.to_csv(summary_path, index=False, columns=cols)
                    else:
                        df_row.to_csv(summary_path, mode="a", index=False, header=False, columns=cols)

if __name__ == "__main__":
    make_summary()