from numba.experimental import structref
from numba import njit, types, typed
from numba.extending import overload_method
from calcite.core.vectors.vector import vector_type, vector
from calcite import constants
import numpy as np

# region ParticleType and Particle
# region Class definitions
@structref.register
class ParticleType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)
    
class Particle(structref.StructRefProxy):
    def __new__(self, mass, charge, spin, position, velocity, data, index):
        instance = structref.StructRefProxy.__new__(self, mass, charge, spin, position, velocity, data, index)
        return instance
    
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
    
    @property
    def index(self):
        return Particle_get_index(self)

    @index.setter
    def index(self, index):
        Particle_set_index(self, index)
    
# endregion
# region Particle methods
# region Fields
@njit(cache=True)
def Particle_get_mass(self):
    return self.mass

@njit(cache=True)
def Particle_get_charge(self):
    return self.charge

@njit(cache=True)
def Particle_get_spin(self):
    return self.spin

@njit(cache=True)
def Particle_get_position(self):
    return self.position

@njit(cache=True)
def Particle_get_velocity(self):
    return self.velocity

@njit(cache=True)
def Particle_get_data(self):
    return self.data

@njit(cache=True)
def Particle_get_index(self):
    return self.index

@njit(cache=True)
def Particle_set_mass(self, mass):
    self.mass = mass

@njit(cache=True)
def Particle_set_charge(self, charge):
    self.charge = charge

@njit(cache=True)
def Particle_set_spin(self, spin):
    self.spin = spin

@njit(cache=True)
def Particle_set_position(self, position):
    self.position = position

@njit(cache=True)
def Particle_set_velocity(self, velocity):
    self.velocity = velocity

@njit(cache=True)
def Particle_set_data(self, data):
    self.data = data

@njit(cache=True)
def Particle_set_index(self, index):
    self.index = index
# endregion
# region Methods

@overload_method(ParticleType, 'momentum')
def Particle_momentum(self):
    def impl(self):
        return self.mass * self.velocity
    return impl

@overload_method(ParticleType, 'kinetic_energy')
def Particle_kinetic_energy(self):
    def impl(self):
        return 0.5 * self.mass * vector.magnitude(self.velocity) ** 2
    return impl

@overload_method(ParticleType, 'relativistic_mass')
def Particle_relativistic_mass(self):
    def impl(self):
        return self.mass / (1 - vector.magnitude(self.velocity) ** 2 / constants.C ** 2) ** 0.5
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
    'velocity', 'data', 'index'
])

particle_type = ParticleType(fields=[
    ('mass', types.float64),
    ('charge', types.float64),
    ('spin', types.float64),
    ('position', vector_type),
    ('velocity', vector_type),
    ('data', types.DictType(types.unicode_type, types.float64)),
    ('index', types.int64)
])
# endregion
# region Particle creation methods
@njit(cache=True)
def electron(n, l, m, spin=None, position=None, velocity=None) -> Particle:
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
    data = typed.Dict.empty(types.unicode_type, types.float64)
    data["type"] = constants.ELECTRON_FLAG
    data["n"] = float(n)
    data["l"] = float(l)
    data["m"] = float(m)
    
    if position is not None:
        if position is vector_type:
            pass
        else:
            position = vector(*position)
    else:
        position = vector(np.nan, np.nan, np.nan)
    
    if velocity is not None:
        if velocity is vector_type:
            pass
        else:
            velocity = vector(*velocity)
    else:
        velocity = vector(np.nan, np.nan, np.nan)

    return Particle(
        constants.ELECTRON_MASS, 
        constants.ELECTRON_CHARGE, 
        constants.ELECTRON_SPIN if spin is None else spin, 
        position, 
        velocity, 
        data, -1
    )

@njit(cache=True)
def particle(mass, charge, spin, position=None, velocity=None) -> Particle:
    """
    Creates a new particle with the given properties.

    Args:
        mass (float): The mass of the particle.
        charge (float): The electric charge of the particle.
        spin (float): The spin of the particle.
        position (list[float] | None, optional): The position of the particle. Defaults to None.
        velocity (list[float] | None, optional): The velocity of the particle. Defaults to None.
        index (int, optional): The index of the particle. Defaults to -1.

    Returns:
        Particle: A new particle object
    """
    data = typed.Dict.empty(types.unicode_type, types.float64)
    
    if position is not None:
        if position is vector_type:
            pass
        else:
            position = vector(*position)
    else:
        position = vector(np.nan, np.nan, np.nan)
    
    if velocity is not None:
        if velocity is vector_type:
            pass
        else:
            velocity = vector(*velocity)
    else:
        velocity = vector(np.nan, np.nan, np.nan)

    return Particle(mass, charge, spin, position, velocity, data, -1)

# endregion