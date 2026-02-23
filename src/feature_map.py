import numpy as np
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap

def create_feature_map(feature_map_name):
    if feature_map_name == "z_feature_map":
        return ZFeatureMap(feature_dimension=3, reps=2)
    elif feature_map_name == "zz_feature_map":
        return ZZFeatureMap(feature_dimension=3, reps=4, entanglement='linear')
    elif feature_map_name == "pauli_feature_map":
        return PauliFeatureMap(feature_dimension=3, reps=2, paulis=['X', 'Y', 'Z'])
    else:
        raise ValueError(f"Unknown feature map name: {feature_map_name}")