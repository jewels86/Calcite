# from typing import
import numpy as np

class Quark:
    """
    A basic class to represent a quark in the Standard Model of particle physics.

    Attributes:
    - flavor (str): the type of quark (up, down, strange, charm, top, bottom)
    - charge (float): the electric charge of the quark
    - mass (float): the mass of the quark in atomic units
    - spin (float): the spin of the quark
    - data (dict): additional data about the quark
    - debug_mode (bool): a flag to enable debug mode
    """
    flavor: str
    """The type of quark (up, down, strange, charm, top, bottom)."""
    charge: float
    """The electric charge of the quark."""
    mass: float
    """The mass of the quark in atomic units."""
    spin: float
    """The spin of the quark."""
    data: dict
    """Additional data about the quark."""
    debug_mode: bool
    """A flag to enable debug mode."""

def up_quark() -> Quark: ...
"""
Creates a new up quark.

Returns:
    Quark: a new up quark object
"""

def down_quark() -> Quark: ...
"""
Creates a new down quark.

Returns:
    Quark: a new down quark object
"""

def strange_quark() -> Quark: ...
"""
Creates a new strange quark.

Returns:
    Quark: a new strange quark object
"""

def charm_quark() -> Quark: ...
"""
Creates a new charm quark.

Returns:
    Quark: a new charm quark object
"""

def top_quark() -> Quark: ...
"""
Creates a new top quark.

Returns:
    Quark: a new top quark object
"""

def bottom_quark() -> Quark: ...
"""
Creates a new bottom quark.

Returns:
    Quark: a new bottom quark object
"""

