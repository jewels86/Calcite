from numba.experimental import structref
from numba import njit, types, typed
from numba.extending import overload_method
from calcite.core.molecules.molecule import Molecule, molecule_type
from calcite.core.vectors.vector import vector, vector_type
import numpy as np

@structref.register
class AggregateType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class Aggregate(structref.StructRefProxy):
    def __new__(cls, molecules, position, velocity, data, debug_mode):
        if isinstance(position, (list, tuple)):
            position = vector(position)
        if isinstance(velocity, (list, tuple)):
            velocity = vector(velocity)
        return structref.StructRefProxy.__new__(cls, molecules, position, velocity, data, debug_mode)

    @property
    def molecules(self):
        return Aggregate_get_molecules(self)

    @molecules.setter
    def molecules(self, molecules):
        Aggregate_set_molecules(self, molecules)

    @property
    def position(self):
        return Aggregate_get_position(self)

    @position.setter
    def position(self, position):
        Aggregate_set_position(self, position)

    @property
    def velocity(self):
        return Aggregate_get_velocity(self)

    @velocity.setter
    def velocity(self, velocity):
        Aggregate_set_velocity(self, velocity)

    @property
    def data(self):
        return Aggregate_get_data(self)

    @data.setter
    def data(self, data):
        Aggregate_set_data(self, data)

    @property
    def debug_mode(self):
        return Aggregate_get_debug_mode(self)

    @debug_mode.setter
    def debug_mode(self, debug_mode):
        Aggregate_set_debug_mode(self, debug_mode)

@njit(cache=True)
def Aggregate_get_molecules(aggregate):
    return aggregate.molecules

@njit(cache=True)
def Aggregate_set_molecules(aggregate, molecules):
    aggregate.molecules = molecules

@njit(cache=True)
def Aggregate_get_position(aggregate):
    return aggregate.position

@njit(cache=True)
def Aggregate_set_position(aggregate, position):
    aggregate.position = position

@njit(cache=True)
def Aggregate_get_velocity(aggregate):
    return aggregate.velocity

@njit(cache=True)
def Aggregate_set_velocity(aggregate, velocity):
    aggregate.velocity = velocity

@njit(cache=True)
def Aggregate_get_data(aggregate):
    return aggregate.data

@njit(cache=True)
def Aggregate_set_data(aggregate, data):
    aggregate.data = data

@njit(cache=True)
def Aggregate_get_debug_mode(aggregate):
    return aggregate.debug_mode

@njit(cache=True)
def Aggregate_set_debug_mode(aggregate, debug_mode):
    aggregate.debug_mode = debug_mode

@overload_method(AggregateType, "add")
def Aggregate_add_molecule(self, molecule):
    def impl(self, molecule):
        self.molecules.append(molecule)
    return impl

@overload_method(AggregateType, "mass")
def Aggregate_total_mass(self):
    def impl(self):
        return sum(molecule.mass() for molecule in self.molecules)
    return impl

@overload_method(AggregateType, "charge")
def Aggregate_total_charge(self):
    def impl(self):
        return sum(molecule.charge() for molecule in self.molecules)
    return impl

@overload_method(AggregateType, "center_of_mass")
def Aggregate_center_of_mass(self):
    def impl(self):
        total_mass = 0.0
        weighted_sum = vector(0.0, 0.0, 0.0)
        for molecule in self.molecules:
            m = molecule.mass()
            total_mass += m
            weighted_sum += molecule.position.xyz * m
        return vector(*(weighted_sum / total_mass)) if total_mass > 0 else vector(0.0, 0.0, 0.0)
    return impl

structref.define_proxy(Aggregate, AggregateType, [
    "molecules", "position", "velocity", "data", "debug_mode"
])

aggregate_type = AggregateType(
    fields=[
        ("molecules", types.ListType(molecule_type)),
        ("position", vector_type),
        ("velocity", vector_type),
        ("data", types.DictType(types.unicode_type, types.float64)), 
        ("debug_mode", types.boolean),
    ]
)

@njit(cache=True)
def aggregate(molecules=None, position=None, velocity=None, data=None, debug_mode=False):
    if molecules is not None:
        _molecules = typed.List.empty_list(molecule_type)
        for mol in molecules:
            _molecules.append(mol)
    else:
        _molecules = typed.List.empty_list(molecule_type)

    position = vector(*position) if position is not None else vector(np.nan, np.nan, np.nan)
    velocity = vector(*velocity) if velocity is not None else vector(np.nan, np.nan, np.nan)
    if data is None:
        data = typed.Dict.empty(key_type=types.unicode_type, value_type=types.unicode_type)
    return Aggregate(_molecules, position, velocity, data, debug_mode)
