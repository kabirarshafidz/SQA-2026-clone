import numpy as np
from qiskit_algorithms.optimizers import COBYLA, SPSA, ADAM

def create_optimizer(name):
    if name == "cobyla":
        return COBYLA(maxiter=100)
    elif name == "spsa":
        return SPSA(maxiter=100)
    elif name == "adam":
        return ADAM(maxiter=40, lr=0.1)
    else:
        raise ValueError(f"Unknown optimizer name: {name}")