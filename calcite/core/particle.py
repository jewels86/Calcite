import numpy as np
from dataclasses import dataclass

@dataclass
class Particle:
    mass: float
    "The mass of the particle in GeV/c^2."
    
    charge: float
    "The charge of the particle in Coulombs."

    momentum: np.ndarray
    "The momentum of the particle in GeV/c."

    energy: float
    "The energy of the particle in GeV."

    velocity: np.ndarray
    "The velocity of the particle in m/s."

    position: np.ndarray
    "The position of the particle in fm."

    positions: np.ndarray
    "The positions of the particle in fm."
    velocities: np.ndarray
    "The velocities of the particle in m/s."
    momenta: np.ndarray
    "The momenta of the particle in GeV/c."
    energies: np.ndarray
    "The energies of the particle in GeV."

    