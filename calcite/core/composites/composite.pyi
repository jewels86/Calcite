import numpy as np

class CompositeParticle:
    """
    A class representing a composite particle made up of quarks.

    Attributes:
    - quarks (list): the list of quarks that make up the composite particle
    - position (np.ndarray): the position of the composite particle in 3D space (may be NaN)
    - velocity (np.ndarray): the velocity of the composite particle in 3D space (may be NaN)
    - data (dict): additional data about the composite particle

    Methods:
    - mass(): returns the mass of the composite particle
    - charge(): returns the electric charge of the composite particle
    - spin(): returns the spin of the composite particle
    - momentum(): returns the momentum of the composite particle
    - bayron_number(): returns the bayron number of the composite particle
    """
    quarks: list
    """The list of quarks that make up the composite particle."""
    position: np.ndarray
    """The position of the composite particle in 3D space (may be NaN)."""
    velocity: np.ndarray
    """The velocity of the composite particle in 3D space (may be NaN)."""
    data: dict
    """Additional data about the composite particle."""

    def mass(self) -> float: ...
    """Returns the mass of the composite particle."""
    def charge(self) -> float: ...
    """Returns the electric charge of the composite particle."""
    def spin(self) -> float: ...
    """Returns the spin of the composite particle."""
    def momentum(self) -> np.ndarray: ...
    """Returns the momentum of the composite particle."""
    def bayron_number(self) -> int: ...
    """Returns the bayron number of the composite particle."""

def proton(position: list[float] | None = None, velocity: list[float] | None = None) -> CompositeParticle: ...
"""
Creates a new proton with the given position and velocity.

Args:
    position (list[float] | None, optional): The position of the proton. Defaults to None.
    velocity (list[float] | None, optional): The velocity of the proton. Defaults to None.

Returns:
    CompositeParticle: A new proton object
"""

def neutron(position: list[float] | None = None, velocity: list[float] | None = None) -> CompositeParticle: ...
"""
Creates a new neutron with the given position and velocity.

Args:
    position (list[float] | None, optional): The position of the neutron. Defaults to None.
    velocity (list[float] | None, optional): The velocity of the neutron. Defaults to None.

Returns:
    CompositeParticle: A new neutron object
"""

