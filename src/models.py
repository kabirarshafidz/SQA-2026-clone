import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, ZFeatureMap, RealAmplitudes, TwoLocal
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

def create_svm_model():
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
    return svm_model

def create_rf_model():
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    return rf_model

def create_vqc_model_1():
    feature_map = ZZFeatureMap(feature_dimension=3, reps=2, entanglement='linear')
    ansatz = RealAmplitudes(num_qubits=3, reps=2, entanglement='linear')
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

    optimizer = COBYLA(maxiter=100)

    vqc = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=optimizer,
        initial_point=np.random.uniform(-1,1, qnn.num_weights)
        # callback
    )

    return vqc

def create_vqc_model_2():
    feature_map = PauliFeatureMap(feature_dimension=3, reps=2, paulis=['X', 'Y', 'Z'])
    ansatz = TwoLocal(num_qubits=3, reps=2, rotation_blocks='ry', entanglement_blocks='cz', entanglement='linear')
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

    optimizer = SPSA(maxiter=100)

    vqc = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=optimizer,
        initial_point=np.random.uniform(-1,1, qnn.num_weights)
        # callback
    )

    return vqc

def create_vqc_model_3():
    feature_map = ZFeatureMap(feature_dimension=3, reps=2)
    ansatz = RealAmplitudes(num_qubits=3, reps=2, entanglement='linear')
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

    optimizer = ADAM(maxiter=40, lr=0.1)

    vqc = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=optimizer,
        initial_point=np.random.uniform(-1,1, qnn.num_weights)
        # callback
    )

    return vqc

def create_vqc_model_4():
    feature_map = ZZFeatureMap(feature_dimension=3, reps=4, entanglement='linear')
    ansatz = RealAmplitudes(num_qubits=3, reps=4, entanglement='linear')
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

    optimizer = COBYLA(maxiter=100)

    vqc = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=optimizer,
        initial_point=np.random.uniform(-1,1, qnn.num_weights)
        # callback
    )

    return vqc

def get_model_by_name(model_name):
    if model_name == "svm":
        return create_svm_model()
    elif model_name == "random_forest":
        return create_rf_model()
    elif model_name == "vqc_1":
        return create_vqc_model_1()
    elif model_name == "vqc_2":
        return create_vqc_model_2()
    elif model_name == "vqc_3":
        return create_vqc_model_3()
    elif model_name == "vqc_4":
        return create_vqc_model_4()
    else:
        raise ValueError(f"Unknown model name: {model_name}")