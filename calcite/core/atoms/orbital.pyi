from typing import List
from calcite.core.particles.particle import Particle

class OrbitalType:
    pass

class Orbital:
    """
    A class representing an atomic orbital.

    Attributes:
    - n (int): the principal quantum number
    - l (int): the azimuthal quantum number
    - m (int): the magnetic quantum number
    - electrons (List[Particle]): the electrons in the orbital
    - debug_mode (bool): flag for debug mode

    Methods:
    - can_add(electron: Particle) -> bool: checks if an electron can be added to the orbital
    - add(electron: Particle) -> bool: adds an electron to the orbital
    """
    n: int
    """The principal quantum number."""
    l: int
    """The azimuthal quantum number."""
    m: int
    """The magnetic quantum number."""
    electrons: List[Particle]
    """The electrons in the orbital."""
    debug_mode: bool
    """Flag for debug mode."""

    def can_add(self, electron: Particle) -> bool: ...
    """Checks if an electron can be added to the orbital."""
    
    def add(self, electron: Particle) -> bool: ...
    """Adds an electron to the orbital."""

orbital_type = OrbitalType