# load configs
import sys
import os
# Ensure repo root is on PYTHONPATH so `import src...` works when running this script directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.experiment import run_one
from src.data import generate_network_traffic_data, generate_network_traffic_data_2
from datetime import datetime

# CLI: python scripts/run_experiments.py --config configs/sweep_anomaly_ratio.json
if len(sys.argv) >= 3 and sys.argv[1] == "--config":
    config_path = sys.argv[2]
else:
    config_path = "configs/sweep_anomaly_ratio.json" # default

with open(config_path, "r") as f:
    sweep_config = json.load(f)

for val in sweep_config["sweep_values"]:
    config = {}

    with open(sweep_config["base_config"], "r") as f:
        config = json.load(f)
        
    config[sweep_config["sweep_key"]] = val
    
    for seed in config["seeds"]:
        np.random.seed(seed)

        X_df, y = generate_network_traffic_data_2(n_samples=config["n_samples"],
                                                  anomaly_ratio=config["anomaly_ratio"],
                                                  label_noise=config["label_noise"])
        
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, stratify=y, random_state=seed)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        data_bundles = {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_test": X_test_scaled,
            "y_test": y_test,
        }

        for model_name in config["model_list"]:
            run_config = dict(config) 
            run_config["model_name"] = model_name
            run_config["seed"] = seed

            print(f"Running: model={model_name}, anomaly_ratio={val}, seed={seed}")

            metrics = run_one(run_config, data_bundles)

            print(f"Completed: model={model_name}, anomaly_ratio={val}, seed={seed}, metrics={metrics}")

            date = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join(
                "experiments",
                sweep_config["sweep_key"],     
                str(val),                      
                f"seed_{seed}",                
                model_name                     
            )
            os.makedirs(exp_dir, exist_ok=True)

            with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            with open(os.path.join(exp_dir, "config.json"), "w") as f:
                json.dump(run_config, f, indent=2)

            cm = metrics.get("confusion_matrix", None)
            if cm is not None:
                with open(os.path.join(exp_dir, "confusion_matrix.csv"), "w") as f:
                    for row in cm:
                        f.write(",".join(str(x) for x in row) + "\n")
