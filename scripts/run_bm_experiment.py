# load configs
import sys
import os
# Ensure repo root is on PYTHONPATH so `import src...` works when running this script directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import json
import pickle
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.experiment import run_one
from src.data import generate_network_traffic_data, generate_network_traffic_data_2
from src.ansatz import create_ansatz
from src.feature_map import create_feature_map
from src.optimizers import create_optimizer
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# Best model: VQC_3
# feature_map = ZFeatureMap(feature_dimension=3, reps=2)
# ansatz = RealAmplitudes(num_qubits=3, reps=2, entanglement='linear')
# estimator = StatevectorEstimator()
# qnn = EstimatorQNN(
#     circuit=qc,
#     input_params=feature_map.parameters,
#     weight_params=ansatz.parameters,
#     estimator=estimator
# )
# optimizer = ADAM(maxiter=40, lr=0.1)
# vqc = NeuralNetworkClassifier(
#     neural_network=qnn,
#     optimizer=optimizer,
#     initial_point=np.random.uniform(-1,1, qnn.num_weights)
#     # callback
# )

if len(sys.argv) >= 3 and sys.argv[1] == "--config":
    config_path = sys.argv[2]
else:
    config_path = "configs/sweep_bm_feature_map.json" # default

with open(config_path, "r") as f:
    sweep_config = json.load(f)

for val in sweep_config["sweep_values"]:
    config = {}

    with open(sweep_config["base_config"], "r") as f:
        config = json.load(f)
    
    run_config = dict(config)

    run_config[sweep_config["sweep_key"]] = val
    
    for seed in run_config["seeds"]:
        np.random.seed(seed)

        if run_config["generator"] == "v1":
            X_df, y = generate_network_traffic_data(n_samples=run_config["n_samples"],
                                                  anomaly_ratio=run_config["anomaly_ratio"],
                                                  label_noise=run_config["label_noise"])
        else:
            X_df, y = generate_network_traffic_data_2(n_samples=run_config["n_samples"],
                                                  anomaly_ratio=run_config["anomaly_ratio"],
                                                  label_noise=run_config["label_noise"])
        
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

        feature_map = create_feature_map(run_config["feature_map"])
        ansatz = create_ansatz(run_config["ansatz"], run_config.get("ansatz_depth", 1))
        qc = QuantumCircuit(3)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)

        estimator = StatevectorEstimator()

        qnn = EstimatorQNN(
            circuit=qc,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=estimator
        )

        optimizer = create_optimizer(run_config["optimizer"])

        vqc = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=optimizer,
            initial_point=np.random.uniform(-1,1, qnn.num_weights)
            # callback
        )

        print(f"Running: model=VQC-3, {sweep_config['sweep_key']}={val}, seed={seed}, data_gen={run_config['generator']}")

        start = time.time()
        model, metrics = run_one(run_config, data_bundles, base_model=vqc)
        end = time.time()

        print(f"Run completed in {end - start:.2f} seconds")
        print(f"Metrics: {metrics}")

        training_time = end - start

        print(f"Completed: model=VQC-3, {sweep_config['sweep_key']}={val}, seed={seed}, metrics={metrics}")
        print(f"Elapsed time: {training_time:.2f} seconds")

        exp_dir = os.path.join(
            "best_model_experiment",
            "experiments_v2" if config["generator"] == "v2" else "experiments_v1",
            sweep_config["sweep_key"],     
            str(val),                      
            f"seed_{seed}"                     
        )
        os.makedirs(exp_dir, exist_ok=True)

        with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            run_config["training_time_seconds"] = training_time
            json.dump(run_config, f, indent=2)

        cm = metrics.get("confusion_matrix", None)
        if cm is not None:
            with open(os.path.join(exp_dir, "confusion_matrix.csv"), "w") as f:
                for row in cm:
                    f.write(",".join(str(x) for x in row) + "\n")

        model.to_dill(f"{exp_dir}/model.dill")

