import numpy as np
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM

def create_optimizer(name):
    if name == "COBYLA":
        return COBYLA(maxiter=100)
    elif name == "SPSA":
        return SPSA(maxiter=100)
    elif name == "ADAM":
        return ADAM(maxiter=40, lr=0.1)
    else:
        raise ValueError(f"Unknown optimizer name: {name}")