from numba.experimental import structref
from numba import njit, types, typed
from numba.extending import overload_method
from calcite.core.quarks.quark import up_quark, down_quark, quark_type
from calcite.core.vectors.vector import vector_type, vector
from calcite import constants
import numpy as np

# region CompositeParticleType and CompositeParticle
# region Class definitions
@structref.register
class CompositeParticleType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)
    
class CompositeParticle(structref.StructRefProxy):
    def __new__(self, quarks, position, velocity, data):
        return structref.StructRefProxy.__new__(self, quarks, position, velocity, data)
    
    @property
    def quarks(self):
        return CompositeParticle_get_quarks(self)
    
    @quarks.setter
    def quarks(self, quarks):
        CompositeParticle_set_quarks(self, quarks)

    @property
    def position(self):
        return CompositeParticle_get_position(self)
    
    @position.setter
    def position(self, position):
        CompositeParticle_set_position(self, position)

    @property
    def velocity(self):
        return CompositeParticle_get_velocity(self)
    
    @velocity.setter
    def velocity(self, velocity):
        CompositeParticle_set_velocity(self, velocity)

    @property
    def data(self):
        CompositeParticle_get_data(self)
    
    @data.setter
    def data(self, data):
        CompositeParticle_set_data(self, data)

# endregion
# region CompositeParticle methods
# region Fields
@njit(cache=True)
def CompositeParticle_get_quarks(self):
    return self.quarks

@njit(cache=True)
def CompositeParticle_set_quarks(self, quarks):
    self.quarks = quarks

@njit(cache=True)
def CompositeParticle_get_position(self):
    return self.position

@njit(cache=True)
def CompositeParticle_set_position(self, position):
    self.position = position

@njit(cache=True)
def CompositeParticle_get_velocity(self):
    return self.velocity

@njit(cache=True)
def CompositeParticle_set_velocity(self, velocity):
    self.velocity = velocity

@njit(cache=True)
def CompositeParticle_get_data(self):
    return self.data

@njit(cache=True)
def CompositeParticle_set_data(self, data):
    self.data = data
# endregion
# region Methods
@overload_method(CompositeParticleType, 'mass')
def CompositeParticle_mass(self):
    def impl(self):
        mass = 0.0
        for quark in self.quarks:
            mass += quark.mass
        return mass
    return impl

@overload_method(CompositeParticleType, 'charge')
def CompositeParticle_charge(self):
    def impl(self):
        charge = 0.0
        for quark in self.quarks:
            charge += quark.charge
        return charge
    return impl

@overload_method(CompositeParticleType, 'spin')
def CompositeParticle_spin(self):
    def impl(self):
        spin = 0.0
        for quark in self.quarks:
            spin += quark.spin
        return spin / len(self.quarks)
    return impl

@overload_method(CompositeParticleType, 'momentum')
def CompositeParticle_momentum(self):
    def impl(self):
        return self.mass * self.velocity
    return impl

@overload_method(CompositeParticleType, 'bayron_number')
def CompositeParticle_bayron_number(self):
    def impl(self):
        return len(self.quarks) // 3
    return impl

# endregion
# endregion
structref.define_proxy(CompositeParticle, CompositeParticleType, [
    'quarks', 'position', 'velocity', 'data'
])
composite_particle_type = CompositeParticleType([
    ('quarks', types.ListType(quark_type)),
    ('position', vector_type),
    ('velocity', vector_type),
    ('data', types.DictType(types.unicode_type, types.unicode_type))
])
# endregion
# region CompositeParticle creation methods
@njit(cache=True)
def proton(position=None, velocity=None):
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    data['type'] = 'proton'
    position = vector(*position) if position is not None else vector(np.nan, np.nan, np.nan)
    velocity = vector(*velocity) if velocity is not None else vector(np.nan, np.nan, np.nan)
    return CompositeParticle(
        typed.List([up_quark(), up_quark(), down_quark()]),
        position,
        velocity,
        data
    )

@njit(cache=True)
def neutron(position=None, velocity=None):
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    data['type'] = 'neutron'
    position = vector(*position) if position is not None else vector(np.nan, np.nan, np.nan)
    velocity = vector(*velocity) if velocity is not None else vector(np.nan, np.nan, np.nan)
    return CompositeParticle(
        typed.List([up_quark(), down_quark(), down_quark()]),
        position,
        velocity,
        data
    )
# endregion