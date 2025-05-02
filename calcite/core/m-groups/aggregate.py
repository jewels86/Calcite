from numba.experimental import structref
from numba import njit, types, typed
from calcite.core.atoms.atom import Atom, atom_type
from calcite.core.vectors.vector import vector, vector_type
from numba.extending import overload_method
import numpy as np

# region MoleculeType
@structref.register
class MoleculeType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)
# endregion
# region Molecule
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
# endregion
#region Fields
@njit(cache=True)
def Molecule_get_atoms(molecule):
    return molecule.atoms

@njit(cache=True)
def Molecule_set_atoms(molecule, atoms):
    molecule.atoms = atoms