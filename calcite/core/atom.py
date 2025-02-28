import numpy as np
from dataclasses import dataclass

@dataclass
class Atom:
    atomic_number: int
    mass: float
    charge: float
    position: np.ndarray
    velocity: np.ndarray
    electron_shells: np.ndarray

    protons: 
