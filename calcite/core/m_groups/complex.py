from numba.experimental import structref
from numba import njit, types, typed
from numba.extending import overload_method
from calcite.core.molecules.molecule import Molecule, molecule_type
from calcite.core.vectors.vector import vector, vector_type, vector_xyz
import numpy as np

@structref.register
class ComplexType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class Complex(structref.StructRefProxy):
    def __new__(cls, molecules, position, velocity, data, debug_mode):
        if isinstance(position, (list, tuple)):
            position = vector(position)
        if isinstance(velocity, (list, tuple)):
            velocity = vector(velocity)
        return structref.StructRefProxy.__new__(cls, molecules, position, velocity, data, debug_mode)

    @property
    def molecules(self):
        return Complex_get_molecules(self)

    @molecules.setter
    def molecules(self, molecules):
        Complex_set_molecules(self, molecules)

    @property
    def position(self):
        return Complex_get_position(self)

    @position.setter
    def position(self, position):
        Complex_set_position(self, position)

    @property
    def velocity(self):
        return Complex_get_velocity(self)

    @velocity.setter
    def velocity(self, velocity):
        Complex_set_velocity(self, velocity)

    @property
    def data(self):
        return Complex_get_data(self)

    @data.setter
    def data(self, data):
        Complex_set_data(self, data)

    @property
    def debug_mode(self):
        return Complex_get_debug_mode(self)

    @debug_mode.setter
    def debug_mode(self, debug_mode):
        Complex_set_debug_mode(self, debug_mode)

@njit(cache=True)
def Complex_get_molecules(complex_):
    return complex_.molecules

@njit(cache=True)
def Complex_set_molecules(complex_, molecules):
    complex_.molecules = molecules

@njit(cache=True)
def Complex_get_position(complex_):
    return complex_.position

@njit(cache=True)
def Complex_set_position(complex_, position):
    complex_.position = position

@njit(cache=True)
def Complex_get_velocity(complex_):
    return complex_.velocity

@njit(cache=True)
def Complex_set_velocity(complex_, velocity):
    complex_.velocity = velocity

@njit(cache=True)
def Complex_get_data(complex_):
    return complex_.data

@njit(cache=True)
def Complex_set_data(complex_, data):
    complex_.data = data

@njit(cache=True)
def Complex_get_debug_mode(complex_):
    return complex_.debug_mode

@njit(cache=True)
def Complex_set_debug_mode(complex_, debug_mode):
    complex_.debug_mode = debug_mode

@overload_method(ComplexType, "add")
def Complex_add_molecule(self, molecule):
    def impl(self, molecule):
        self.molecules.append(molecule)
    return impl

@overload_method(ComplexType, "mass")
def Complex_total_mass(self):
    def impl(self):
        return sum(molecule.mass() for molecule in self.molecules)
    return impl

@overload_method(ComplexType, "charge")
def Complex_total_charge(self):
    def impl(self):
        return sum(molecule.charge() for molecule in self.molecules)
    return impl

@overload_method(ComplexType, "center_of_mass")
def Complex_center_of_mass(self):
    def impl(self):
        total_mass = 0.0
        weighted_sum = np.array([0.0, 0.0, 0.0])
        for molecule in self.molecules:
            m = molecule.mass()
            total_mass += m
            weighted_sum += molecule.position.xyz() * m
        return vector_xyz((weighted_sum / total_mass)) if total_mass > 0 else vector(0.0, 0.0, 0.0)
    return impl

@overload_method(ComplexType, "intermolecular_bonds")
def Complex_intermolecular_bonds(self):
    def impl(self):
        intermolecular_bonds = []
        for molecule in self.molecules:
            for bond in molecule.bonds():
                if bond[0] in self.molecules:
                    intermolecular_bonds.append(bond)
        intermolecular_bonds = typed.List(intermolecular_bonds)
        return intermolecular_bonds
    return impl

@overload_method(ComplexType, "structure")
def Complex_structure(self):
    def impl(self):
        structure = {
            "molecules": len(self.molecules),
            "intermolecular_bonds": self.intermolecular_bonds(),
        }
        return structure
    return impl

structref.define_proxy(Complex, ComplexType, [
    "molecules", "position", "velocity", "data", "debug_mode"
])

complex_type = ComplexType(
    fields=[
        ("molecules", types.ListType(molecule_type)),
        ("position", vector_type),
        ("velocity", vector_type),
        ("data", types.DictType(types.unicode_type, types.float64)),
        ("debug_mode", types.boolean),
    ]
)

@njit(cache=True)
def complex_(molecules=None, position=None, velocity=None, data=None, debug_mode=False):
    if molecules is not None:
        _molecules = typed.List.empty_list(molecule_type)
        for mol in molecules:
            _molecules.append(mol)
    else:
        _molecules = typed.List.empty_list(molecule_type)

    position = vector(*position) if position is not None else vector(np.nan, np.nan, np.nan)
    velocity = vector(*velocity) if velocity is not None else vector(np.nan, np.nan, np.nan)
    if data is None:
        data = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    return Complex(_molecules, position, velocity, data, debug_mode)

