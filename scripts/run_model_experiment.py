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
from datetime import datetime

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
