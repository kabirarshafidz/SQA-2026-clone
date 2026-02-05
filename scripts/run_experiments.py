# load configs
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.experiment import run_one
from src.data import generate_network_traffic_data, generate_network_traffic_data_2

with open("configs/sweep_anomaly_ratio.json", "r") as f:
    sweep_config = json.load(f)

# the loop
    # call the experiment runner from experiment.py

for val in sweep_config["sweep_values"]:
    config = {}

    with open(sweep_config["base_config", "r"]) as f:
        config = json.load(f)
        config[sweep_config["sweep_key"]] = val
    
    for seed in config["seeds"]:
        np.random.seed(seed)

        X_df, y = generate_network_traffic_data_2(n_samples=config["n_samples"],
                                                  anomaly_ratio=config["anomaly_ratio"],
                                                  label_noise=config["label_noise"])
        
        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        data_bundles = {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_test": X_test_scaled,
            "y_test": y_test
        }

        for model_name in config["model_list"]:
            config["model_name"] = model_name
            metrics = run_one(config, data_bundles)
            print(f"Completed: model={model_name}, anomaly_ratio={val}, seed={seed}, metrics={metrics}")

