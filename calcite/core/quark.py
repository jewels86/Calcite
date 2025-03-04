from numba import float64, int64, types, njit, typeof, typed
from numba.experimental import jitclass
from calcite.constants import COLORS

spec = [
    ('type', types.string),
    ('charge', float64),
    ('mass', float64),
    ('spin', float64),
    ('color', types.string),
    ('index', int64),
    ('data', types.DictType(types.string, types.float64)),
    ('debug_mode', types.boolean)
]

@jitclass(spec)
class Quark:
    """
    A basic class to represent a quark in the Standard Model of particle physics.

    Attributes:
    - type (str): the type of quark (up, down, strange, charm, top, bottom)
    - charge (float): the electric charge of the quark
    - mass (float): the mass of the quark in MeV/c^2
    - spin (float): the spin of the quark
    - data (dict): additional data about the quark
    - debug_mode (bool): a flag to enable debug mode
    """
    def __init__(self, type: str, charge: float, mass: float, spin: float, data: dict):
        """
        Initialize a Quark object with the given type, charge, mass, spin, and data.
        
        Args:
        - type (str): the type of quark (up, down, strange, charm, top, bottom)
        - charge (float): the electric charge of the quark
        - mass (float): the mass of the quark in MeV/c^2
        - spin (float): the spin of the quark
        - data (dict): additional data about the quark

        Returns:
            Quark: a Quark object with the specified attributes
        """
        self.type: str = type
        "The type of quark (up, down, strange, charm, top, bottom)."
        self.charge: float = charge
        "The electric charge of the quark."
        self.mass: float = mass
        "The mass of the quark in atomic units."
        self.spin: float = spin
        "The spin of the quark."
        self.data = data
        "Additional data about the quark."
        self.debug_mode: bool = False

@njit
def up_quark() -> Quark:
    """
    Creates a new up quark.

    Returns:
        Quark: a new up quark object
    """
    data = typed.Dict.empty(types.unicode_type, float64)
    return Quark('up', 2/3, 0.0022, 0.5, data)

@njit
def down_quark() -> Quark:
    """
    Creates a new down quark.

    Returns:
        Quark: a new down quark object
    """
    data = typed.Dict.empty(types.unicode_type, float64)
    return Quark('down', -1/3, 0.0047, 0.5, data)

@njit
def strange_quark() -> Quark:
    """
    Creates a new strange quark.

    Returns:
        Quark: a new strange quark object
    """
    data = typed.Dict.empty(types.unicode_type, float64)
    return Quark('strange', -1/3, 0.093, 0.5, data)

@njit
def charm_quark() -> Quark:
    """
    Creates a new charm quark.

    Returns:
        Quark: 
    """
    data = typed.Dict.empty(types.unicode_type, float64)
    return Quark('charm', 2/3, 1.27, 0.5, data)

@njit
def top_quark() -> Quark:
    """
    Creates a new top quark.

    Returns:
        Quark: a new top quark object
    """
    data = typed.Dict.empty(types.unicode_type, float64)
    return Quark('top', 2/3, 173.1, 0.5, data)

@njit
def bottom_quark() -> Quark:
    """
    Creates a new bottom quark.

    Returns:
        Quark: a new bottom quark object
    """
    data = typed.Dict.empty(types.unicode_type, float64)
    return Quark('bottom', -1/3, 4.18, 0.5, data)

quark_type = typeof(up_quark())