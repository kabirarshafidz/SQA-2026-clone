import numpy as np
from qiskit.circuit.library import RealAmplitudes, TwoLocal

def create_ansatz(ansatz_name, depth):
    if ansatz_name == "real_amplitudes":
        return RealAmplitudes(num_qubits=3, reps=depth, entanglement='linear')
    elif ansatz_name == "two_local":
        return TwoLocal(num_qubits=3, reps=depth, entanglement='linear')
    else:
        raise ValueError(f"Unknown ansatz name: {ansatz_name}")