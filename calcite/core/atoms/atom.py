from numba.experimental import structref
from numba import njit, types, typed
from numba.extending import overload_method
from calcite.formulas import magnitude
from calcite.core.particles.particle import Particle
from calcite.core.atoms.orbital import Orbital
import numpy as np

# region AtomType and Atom
# region Class definitions
@structref.register
class AtomType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class Atom(structref.StructRefProxy):
    def __new__(cls, n_protons, n_neutrons, n_electrons, position, velocity, data):
        return structref.StructRefProxy.__new__(cls, n_protons, n_neutrons, n_electrons, position, velocity, data)
    
    @property
    def n_protons(self):
        return Atom_get_n_protons(self)
    
    @n_protons.setter
    def n_protons(self, value):
        Atom_set_n_protons(self, value)

    @property
    def n_neutrons(self):
        return Atom_get_n_neutrons(self)

    @n_neutrons.setter
    def n_neutrons(self, value):
        Atom_set_n_neutrons(self, value)

    @property
    def n_electrons(self):
        return Atom_get_n_electrons(self)

    @n_electrons.setter
    def n_electrons(self, value):
        Atom_set_n_electrons(self, value)

    @property
    def position(self):
        return Atom_get_position(self)

    @position.setter
    def position(self, value):
        Atom_set_position(self, value)

    @property
    def velocity(self):
        return Atom_get_velocity(self)

    @velocity.setter
    def velocity(self, value):
        Atom_set_velocity(self, value)

    @property
    def data(self):
        return Atom_get_data(self)

    @data.setter
    def data(self, value):
        Atom_set_data(self, value)

    @property
    def covalent_bonds(self):
        return Atom_get_covalent_bonds(self)

    @covalent_bonds.setter
    def covalent_bonds(self, value):
        Atom_set_covalent_bonds(self, value)

    @property
    def ionic_bonds(self):
        return Atom_get_ionic_bonds(self)

    @ionic_bonds.setter
    def ionic_bonds(self, value):
        Atom_set_ionic_bonds(self, value)

    @property
    def index(self):
        return Atom_get_index(self)

    @index.setter
    def index(self, value):
        Atom_set_index(self, value)

# endregion
# region Atom methods
# region Fields
@njit
def Atom_get_n_protons(atom):
    return atom.n_protons

@njit
def Atom_set_n_protons(atom, value):
    atom.n_protons = value

@njit
def Atom_get_n_neutrons(atom):
    return atom.n_neutrons

@njit
def Atom_set_n_neutrons(atom, value):
    atom.n_neutrons = value

@njit
def Atom_get_n_electrons(atom):
    return atom.n_electrons

@njit
def Atom_set_n_electrons(atom, value):
    atom.n_electrons = value

@njit
def Atom_get_position(atom):
    return atom.position

@njit
def Atom_set_position(atom, value):
    atom.position = value

@njit
def Atom_get_velocity(atom):
    return atom.velocity

@njit
def Atom_set_velocity(atom, value):
    atom.velocity = value

@njit
def Atom_get_data(atom):
    return atom.data

@njit
def Atom_set_data(atom, value):
    atom.data = value

@njit
def Atom_get_covalent_bonds(atom):
    return atom.covalent_bonds

@njit
def Atom_set_covalent_bonds(atom, value):
    atom.covalent_bonds = value

@njit
def Atom_get_ionic_bonds(atom):
    return atom.ionic_bonds

@njit
def Atom_set_ionic_bonds(atom, value):
    atom.ionic_bonds = value

@njit
def Atom_get_index(atom):
    return atom.index

@njit
def Atom_set_index(atom, value):
    atom.index = value
# endregion
# region Methods
@overload_method(AtomType, 'init')
def Atom_init(self):
    def impl(self):
        pass
    return impl