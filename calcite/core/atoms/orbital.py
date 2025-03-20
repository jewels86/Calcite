from numba.experimental import structref
from numba import njit, types, typed
from numba.extending import overload_method
from calcite.formulas import magnitude
from calcite.core.particles.particle import Particle
import numpy as np

# region OrbitalType and Orbital
# region Class definitions
@structref.register
class OrbitalType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)
    
class Orbital(structref.StructRefProxy):
    def __new__(cls, n, l, m, electrons, debug_mode=False):
        return structref.StructRefProxy.__new__(cls, n, l, m, electrons, debug_mode)
    
    @property
    def n(self):
        return Orbital_get_n(self)
    
    @n.setter
    def n(self, n):
        Orbital_set_n(self, n)
    
    @property
    def l(self):
        return Orbital_get_l(self)
    
    @l.setter
    def l(self, l):
        Orbital_set_l(self, l)

    @property
    def m(self):
        return Orbital_get_m(self)
    
    @m.setter
    def m(self, m):
        Orbital_set_m(self, m)

    @property
    def electrons(self):
        return Orbital_get_electrons(self)
    
    @electrons.setter
    def electrons(self, electrons):
        Orbital_set_electrons(self, electrons)

    @property
    def debug_mode(self):
        return Orbital_get_debug_mode(self)
    
    @debug_mode.setter
    def debug_mode(self, debug_mode):
        Orbital_set_debug_mode(self, debug_mode)


# endregion
# region Orbital fields
@njit
def Orbital_get_n(self):
    return self.n

@njit
def Orbital_set_n(self, n):
    self.n = n

@njit
def Orbital_get_l(self):
    return self.l

@njit
def Orbital_set_l(self, l):
    self.l = l

@njit
def Orbital_get_m(self):
    return self.m

@njit
def Orbital_set_m(self, m):
    self.m = m

@njit
def Orbital_get_electrons(self):
    return self.electrons

@njit
def Orbital_set_electrons(self, electrons):
    self.electrons = electrons

@njit
def Orbital_get_debug_mode(self):
    return self.debug_mode

@njit
def Orbital_set_debug_mode(self, debug_mode):
    self.debug_mode = debug_mode

# endregion
# region Orbital methods
@njit
def can_add(self, electron):
    if self.debug_mode: print(f"Orbital.can_add: Checking if electron with spin ({'up' if electron.spin == 0.5 else 'down'}) can be added to orbital ({self.n}, {self.l}, {self.m})")
    if self.debug_mode: print(f"Orbital.can_add: Current electrons in orbital: {len(self.electrons)} ({('up' if self.electrons[0].spin == 0.5 else 'down') if len(self.electrons) > 0 else 'None'\
                                                                                                        }, {('up' if self.electrons[1].spin == 0.5 else 'down') if len(self.electrons) > 1 else 'None'})")
    can_add = len(self.electrons) < 2 and (len(self.electrons) == 0 or electron.spin != self.electrons[0].spin)
    if self.debug_mode: print(f"Orbital.can_add: can_add: {can_add}")
    return can_add

@njit
def add(self, electron):
    if self.debug_mode: 
        print(f"Orbital.add: Attempting to add electron with spin ({'up' if electron.spin == 0.5 else 'down'}) to orbital ({self.n}, {self.l}, {self.m})")
    if self.can_add(electron):
        self.electrons.append(electron)
        if self.debug_mode: 
            print(f"Orbital.add: Added electron with spin ({'up' if electron.spin == 0.5 else 'down'}) to orbital ({self.n}, {self.l}, {self.m})")
        return True
    print(f"Orbital.add: Could not add electron with spin ({'up' if electron.spin == 0.5 else 'down'}) to orbital ({self.n}, {self.l}, {self.m})")
    return False

@overload_method(OrbitalType, 'add')
def Orbital_add(self):
    def impl(self, electron):
        return add(self, electron)
    return impl

@overload_method(OrbitalType, 'can_add')
def Orbital_can_add(self):
    def impl(self, electron):
        return can_add(self, electron)
    return impl
# endregion
structref.define_proxy(Orbital, OrbitalType, ['n', 'l', 'm', 'electrons', 'debug_mode'])
orbital_type = OrbitalType([
    ('n', types.int64),
    ('l', types.int64),
    ('m', types.int64),
    ('electrons', types.ListType(Particle)),
    ('debug_mode', types.boolean)
])
# endregion

# region Orbital creation functions
@njit
def orbital(n, l, m, electrons=None, debug_mode=False):
    if electrons is None:
        electrons = typed.List.empty_list(Particle)
    return Orbital(n, l, m, electrons, debug_mode)
# endregion