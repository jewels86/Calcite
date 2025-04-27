#from typing import 
import numpy as np

class ParticleType:
    pass

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

def electron(n: int, l: int, m: int, spin: float = None,
            position: list[float] | None = None, velocity: list[float] = None) -> Particle: ...
"""
Creates a new electron with the given quantum numbers.

Args:
    n (int): The principal quantum number.
    l (int): The azimuthal quantum number.
    m (int): The magnetic quantum number.
    spin (float, optional): The spin of the electron. Defaults to None.
    position (list[float] | None, optional): The position of the electron. Defaults to None.
    velocity (list[float], optional): The velocity of the electron. Defaults to None.

Returns:
    Particle: A new electron object
"""

def particle(mass: float, charge: float, spin: float, position: list[float] | None = None, 
             velocity: list[float] | None = None) -> Particle: ...
"""
Creates a new particle with the given properties.

Args:
    mass (float): The mass of the particle.
    charge (float): The electric charge of the particle.
    spin (float): The spin of the particle.
    position (list[float] | None, optional): The position of the particle. Defaults to None.
    velocity (list[float] | None, optional): The velocity of the particle. Defaults to None.

Returns:
    Particle: A new particle object
"""

particle_type = ParticleType
"""The type of a particle object."""