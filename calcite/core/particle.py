from numba.experimental import structref
from numba import njit, types, typed, float64
from numba.extending import overload_method
from calcite.formulas import magnitude
from calcite import constants
import numpy as np

# region ParticleType and Particle
# region Class definitions
@structref.register
class ParticleType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)
    
class Particle(structref.StructRefProxy):
    def __new__(self, mass, charge, spin, position, velocity, data):
        return structref.StructRefProxy.__new__(self, mass, charge, spin, position, velocity, data)
    
    @property
    def mass(self):
        return Particle_get_mass(self)
    
    @mass.setter
    def mass(self, mass):
        Particle_set_mass(self, mass)
    
    @property
    def charge(self):
        return Particle_get_charge(self)

    @charge.setter
    def charge(self, charge):
        Particle_set_charge(self, charge)
    
    @property
    def spin(self):
        return Particle_get_spin(self)
    
    @spin.setter
    def spin(self, spin):
        Particle_set_spin(self, spin)
    
    @property
    def position(self):
        return Particle_get_position(self)
    
    @position.setter
    def position(self, position):
        Particle_set_position(self, position)
    
    @property
    def velocity(self):
        return Particle_get_velocity(self)

    @velocity.setter
    def velocity(self, velocity):
        Particle_set_velocity(self, velocity)
    
    @property
    def data(self):
        return Particle_get_data(self)
    
    @data.setter
    def data(self, data):
        Particle_set_data(self, data)
    
# endregion
# region Particle methods
# region Fields
@njit
def Particle_get_mass(self):
    return self.mass

@njit
def Particle_get_charge(self):
    return self.charge

@njit
def Particle_get_spin(self):
    return self.spin

@njit
def Particle_get_position(self):
    return self.position

@njit
def Particle_get_velocity(self):
    return self.velocity

@njit
def Particle_get_data(self):
    return self.data

@njit
def Particle_set_mass(self, mass):
    self.mass = mass

@njit
def Particle_set_charge(self, charge):
    self.charge = charge

@njit
def Particle_set_spin(self, spin):
    self.spin = spin

@njit
def Particle_set_position(self, position):
    self.position = position

@njit
def Particle_set_velocity(self, velocity):
    self.velocity = velocity

@njit
def Particle_set_data(self, data):
    self.data = data
# endregion
# region Class methods

@overload_method(ParticleType, 'momentum')
def Particle_momentum(self):
    def impl(self):
        return self.mass * self.velocity
    return impl

@overload_method(ParticleType, 'kinetic_energy')
def Particle_kinetic_energy(self):
    def impl(self):
        return 0.5 * self.mass * magnitude(self.velocity) ** 2
    return impl

@overload_method(ParticleType, 'relativistic_mass')
def Particle_relativistic_mass(self):
    def impl(self):
        return self.mass / (1 - magnitude(self.velocity) ** 2 / constants.C ** 2) ** 0.5
    return impl

@overload_method(ParticleType, 'energy')
def Particle_energy(self):
    def impl(self):
        return self.relativistic_mass() * constants.C ** 2
    return impl

# endregion
# endregion

structref.define_proxy(Particle, ParticleType, [
    'mass', 'charge', 
    'spin', 'position', 
    'velocity', 'data'
])
# endregion
# region Particle creation methods
@njit
def electron(n: int, l: int, m: int, position: list[float] | None = None, 
             velocity: list[float] = None) -> Particle:
    """
    Creates a new electron with the given quantum numbers.

    Args:
        n (int): The principal quantum number.
        l (int): The azimuthal quantum number.
        m (int): The magnetic quantum number.
        position (list[float] | None, optional): The position of the electron. Defaults to None.
        velocity (list[float], optional): The velocity of the electron. Defaults to None.
        energy (float, optional): The energy of the electron. Defaults to -1.

    Returns:
        Particle: A new electron object
    """
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    data["type"] = "electron"
    data["n"] = str(n)
    data["l"] = str(l)
    data["m"] = str(m)
    position = np.array(position, dtype=np.float64) if position is not None else None
    velocity = np.array(velocity, dtype=np.float64) if velocity is not None else None

    return Particle(
        constants.ELECTRON_MASS, 
        constants.ELECTRON_CHARGE, 
        constants.ELECTRON_SPIN, 
        position, 
        velocity, 
        data
    )

# endregion