from numba.experimental import structref
from numba import njit, types, typed
from calcite.core.atoms.atom import Atom, atom_type
from calcite.core.vectors.vector import vector, vector_type
from numba.extending import overload_method
import numpy as np

@structref.register
class MoleculeType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class Molecule(structref.StructRefProxy):
    def __new__(cls, atoms, position, velocity, debug_mode):
        if isinstance(position, (list, tuple)):
            position = vector(position)
        if isinstance(velocity, (list, tuple)):
            velocity = vector(velocity)
        return structref.StructRefProxy.__new__(cls, atoms, position, velocity, debug_mode)

    @property
    def atoms(self):
        return Molecule_get_atoms(self)

    @atoms.setter
    def atoms(self, atoms):
        Molecule_set_atoms(self, atoms)

    @property
    def position(self):
        return Molecule_get_position(self)

    @position.setter
    def position(self, position):
        Molecule_set_position(self, position)

    @property
    def velocity(self):
        return Molecule_get_velocity(self)

    @velocity.setter
    def velocity(self, velocity):
        Molecule_set_velocity(self, velocity)

    @property
    def debug_mode(self):
        return Molecule_get_debug_mode(self)

    @debug_mode.setter
    def debug_mode(self, debug_mode):
        Molecule_set_debug_mode(self, debug_mode)

@njit(cache=True)
def Molecule_get_atoms(molecule):
    return molecule.atoms

@njit(cache=True)
def Molecule_set_atoms(molecule, atoms):
    molecule.atoms = atoms

@njit(cache=True)
def Molecule_get_position(molecule):
    return molecule.position

@njit(cache=True)
def Molecule_set_position(molecule, position):
    molecule.position = position

@njit(cache=True)
def Molecule_get_velocity(molecule):
    return molecule.velocity

@njit(cache=True)
def Molecule_set_velocity(molecule, velocity):
    molecule.velocity = velocity

@njit(cache=True)
def Molecule_get_debug_mode(molecule):
    return molecule.debug_mode

@njit(cache=True)
def Molecule_set_debug_mode(molecule, debug_mode):
    molecule.debug_mode = debug_mode

@overload_method(MoleculeType, "add")
def Molecule_add_atom(self, atom):
    def impl(self, atom):
        self.atoms.append(atom)
    return impl

@overload_method(MoleculeType, "mass")
def Molecule_mass(self):
    def impl(self):
        mass = 0.0
        for atom in self.atoms:
            mass += atom.mass()
        return mass
    return impl

@overload_method(MoleculeType, "bonds")
def Molecule_get_all_bonds(self):
    def impl(self):
        bonds = []
        for atom in self.atoms:
            bonds.extend(atom.covalent_bonds)
        return bonds
    return impl

@overload_method(MoleculeType, "charge")
def Molecule_charge(self):
    def impl(self):
        total_charge = 0.0
        for atom in self.atoms:
            total_charge += atom.charge()
        return total_charge
    return impl

structref.define_proxy(Molecule, MoleculeType, [
    "atoms", "position", "velocity", "debug_mode"
])

molecule_type = MoleculeType(
    fields=[
        ("atoms", types.ListType(atom_type)),
        ("position", vector_type),
        ("velocity", vector_type),
        ("debug_mode", types.boolean),
    ]
)

@njit(cache=True)
def molecule(atoms=None, position=None, velocity=None, debug_mode=False):
    if atoms is not None:
        _atoms = typed.List.empty_list(atom_type)
        for atom in atoms:
            _atoms.append(atom)
    else:
        _atoms = typed.List.empty_list(atom_type)

    position = vector(*position) if position is not None else vector(np.nan, np.nan, np.nan)
    velocity = vector(*velocity) if velocity is not None else vector(np.nan, np.nan, np.nan)
    return Molecule(_atoms, position, velocity, debug_mode)
