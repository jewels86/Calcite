from numba.experimental import structref
from numba import njit, types, typed, float64
from numba.extending import overload
from calcite.formulas import magnitude
from calcite import constants

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
    
    @property.setter
    def mass(self, mass):
        Particle_set_mass(self, mass)
    
    @property
    def charge(self):
        return Particle_get_charge(self)

    @property.setter
    def charge(self, charge):
        Particle_set_charge(self, charge)
    
    @property
    def spin(self):
        return Particle_get_spin(self)
    
    @property.setter
    def spin(self, spin):
        Particle_set_spin(self, spin)
    
    @property
    def position(self):
        return Particle_get_position(self)
    
    @property.setter
    def position(self, position):
        Particle_set_position(self, position)
    
    @property
    def velocity(self):
        return Particle_get_velocity(self)

    @property.setter
    def velocity(self, velocity):
        Particle_set_velocity(self, velocity)
    
    @property
    def data(self):
        return Particle_get_data(self)
    
    @property.setter
    def data(self, data):
        Particle_set_data(self, data)

    @property
    def momentum(self):
        return momentum(self)
    
    @property
    def kinetic_energy(self):
        return kinetic_energy(self)
    
    @property
    def relativistic_mass(self):
        return relativistic_mass(self)
    
    @property
    def energy(self):
        return energy(self)
    
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
# region Properties
@njit
def momentum(self):
    return self.mass * self.velocity

@njit
def kinetic_energy(self):
    return 0.5 * self.mass * magnitude(self.velocity)**2

@njit
def relativistic_mass(self):
    return self.mass / (1 - magnitude(self.velocity)**2)**0.5

@njit
def energy(self):
    return self.relativistic_mass() * magnitude(self.velocity)


structref.define_proxy(Particle, ParticleType, ['mass', 'charge', 'spin', 'position', 'velocity', 'data'])
# endregion
# endregion

# region Particle creation methods
@njit
def electron(n: int, l: int, m: int, position: list[float] | None = None, 
             velocity: list[float] = None) -> Particle:
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    data["type"] = "electron"
    data["n"] = str(n)
    data["l"] = str(l)
    data["m"] = str(m)

    return Particle(
        constants.ELECTRON_MASS, 
        constants.ELECTRON_CHARGE, 
        constants.ELECTRON_SPIN, 
        position, 
        velocity, 
        data
    )
