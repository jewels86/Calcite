from numba.experimental import structref
from numba import njit, types, typed, float64
from numba.extending import overload_method
from numba.core.errors import TypingError

# region QuarkType and Quark
# region Class definitions
@structref.register
class QuarkType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class Quark(structref.StructRefProxy):
    def __new__(cls, flavor: str, charge: float, mass: float, spin: float, data: dict, index: int):
        instance = structref.StructRefProxy.__new__(cls, flavor, charge, mass, spin, data, index)
        return instance
    
    @property
    def flavor(self) -> str:
        return Quark_get_flavor(self)
    
    @flavor.setter
    def flavor(self, value: str):
        Quark_set_flavor(self, value)
    
    @property
    def charge(self) -> float:
        return Quark_get_charge(self)
    
    @charge.setter
    def charge(self, value: float):
        Quark_set_charge(self, value)
    
    @property
    def mass(self) -> float:
        return Quark_get_mass(self)
    
    @mass.setter
    def mass(self, value: float):
        Quark_set_mass(self, value)
    
    @property
    def spin(self) -> float:
        return Quark_get_spin(self)
    
    @spin.setter
    def spin(self, value: float):
        Quark_set_spin(self, value)

    @property
    def data(self) -> dict:
        return Quark_get_data(self)

    @data.setter
    def data(self, value: dict):
        Quark_set_data(self, value)

    @property
    def index(self) -> int:
        return Quark_get_index(self)

    @index.setter
    def index(self, value: int):
        Quark_set_index(self, value)

# endregion
# region Quark methods

@njit(cache=True)
def Quark_get_flavor(self):
    return self.flavor

@njit(cache=True)
def Quark_get_charge(self):
    return self.charge

@njit(cache=True)
def Quark_get_mass(self):
    return self.mass

@njit(cache=True)
def Quark_get_spin(self):
    return self.spin

@njit(cache=True)
def Quark_get_data(self):
    return self.data

@njit(cache=True)
def Quark_get_index(self):
    return self.index

@njit(cache=True)
def Quark_set_flavor(self, value: str):
    self.flavor = value

@njit(cache=True)
def Quark_set_charge(self, value: float):
    self.charge = value

@njit(cache=True)
def Quark_set_mass(self, value: float):
    self.mass = value

@njit(cache=True)
def Quark_set_spin(self, value: float):
    self.spin = value

@njit(cache=True)
def Quark_set_data(self, value: dict):
    self.data = value

@njit(cache=True)
def Quark_set_index(self, value: int):
    self.index = value
# endregion
# endregion
structref.define_proxy(Quark, QuarkType, ['flavor', 'charge', 'mass', 'spin', 'data', 'index'])
quark_type = QuarkType([
    ('flavor', types.unicode_type),
    ('charge', types.float64),
    ('mass', types.float64),
    ('spin', types.float64),
    ('data', types.DictType(types.unicode_type, types.unicode_type)),
    ('index', types.int64)
])
#region Quark creation methods
@njit(cache=True)
def up_quark() -> Quark:
    """
    Creates a new up quark.

    Returns:
        Quark: a new up quark object
    """
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    return Quark('up', 2/3, 0.0022, 0.5, data, -1)

@njit(cache=True)
def down_quark() -> Quark:
    """
    Creates a new down quark.

    Returns:
        Quark: a new down quark object
    """
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    return Quark('down', -1/3, 0.0047, 0.5, data, -1)

@njit(cache=True)
def strange_quark() -> Quark:
    """
    Creates a new strange quark.

    Returns:
        Quark: a new strange quark object
    """
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    return Quark('strange', -1/3, 0.093, 0.5, data, -1)

@njit(cache=True)
def charm_quark() -> Quark:
    """
    Creates a new charm quark.

    Returns:
        Quark: 
    """
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    return Quark('charm', 2/3, 1.27, 0.5, data, -1)

@njit(cache=True)
def top_quark() -> Quark:
    """
    Creates a new top quark.

    Returns:
        Quark: a new top quark object
    """
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    return Quark('top', 2/3, 173.1, 0.5, data, -1)

@njit(cache=True)
def bottom_quark() -> Quark:
    """
    Creates a new bottom quark.

    Returns:
        Quark: a new bottom quark object
    """
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    return Quark('bottom', -1/3, 4.18, 0.5, data, -1)

#endregion