from numba.experimental import structref
from numba import njit, types, typed, float64
from numba.extending import overload

# region QuarkType and Quark
# region Class definitions
@structref.register
class QuarkType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class Quark(structref.StructRefProxy):
    def __new__(cls, flavor: str, charge: float, mass: float, spin: float, data: dict):
        return structref.StructRefProxy.__new__(cls, flavor, charge, mass, spin, data)
    
    @property
    def flavor(self) -> str:
        return Quark_get_flavor(self)
    
    @property
    def charge(self) -> float:
        return Quark_get_charge(self)
    
    @property
    def mass(self) -> float:
        return Quark_get_mass(self)
    
    @property
    def spin(self) -> float:
        return Quark_get_spin(self)

    @property
    def data(self) -> dict:
        return Quark_get_data(self)

# endregion
# region Quark methods

@njit
def Quark_get_flavor(self):
    return self.flavor

@njit
def Quark_get_charge(self):
    return self.charge

@njit
def Quark_get_mass(self):
    return self.mass

@njit
def Quark_get_spin(self):
    return self.spin

@njit
def Quark_get_data(self):
    return self.data

structref.define_proxy(Quark, QuarkType, ['flavor', 'charge', 'mass', 'spin', 'data'])
# endregion
# endregion

#region Quark creation methods
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

#endregion