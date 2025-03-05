#from typing import 
import numpy as np

class Particle:
    """
    A class representing a particle in the Standard Model of particle physics.

    Attributes:
    - mass (float): the mass of the particle in atomic units
    - charge (float): the electric charge of the particle
    - spin (float): the spin of the particle
    - position (np.ndarray): the position of the particle in 3D space (may be NaN)
    - velocity (np.ndarray): the velocity of the particle in 3D space (may be NaN)
    - data (dict): additional data about the particle

    Methods:
    - momentum(): returns the momentum of the particle
    - energy(): returns the total energy of the particle
    - kinetic_energy(): returns the kinetic energy of the particle
    - relativistic_mass(): returns the relativistic mass of the particle
    """
    mass: float
    """The mass of the particle in atomic units."""
    charge: float
    """The electric charge of the particle."""
    spin: float
    """The spin of the particle."""
    position: np.ndarray
    """The position of the particle in 3D space (may be NaN)."""
    velocity: np.ndarray
    """The velocity of the particle in 3D space (may be NaN)."""
    data: dict
    """Additional data about the particle."""

    def momentum(self) -> np.ndarray: ...
    """Returns the momentum of the particle."""
    def energy(self) -> float: ...
    """Returns the total energy of the particle."""
    def kinetic_energy(self) -> float: ...
    """Returns the kinetic energy of the particle."""
    def relativistic_mass(self) -> float: ...
    """Returns the relativistic mass of the particle."""