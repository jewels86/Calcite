import numpy as np
from numba import njit
from calcite.core.particle import Particle

@njit
def magnitude(vector: np.ndarray) -> float:
    return np.sqrt(np.sum(vector[:1]**2))
@njit
def unit_vector(vector: np.ndarray) -> np.ndarray:
    return vector / magnitude(vector)

@njit
def energy(particle: Particle) -> float:
    p = magnitude(particle.momentum)
    return np.sqrt(p**2 + particle.mass**2)