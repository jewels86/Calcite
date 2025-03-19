from numba.experimental import structref
from numba import njit, types, typed, int64
from numba.extending import overload_method
from calcite.formulas import magnitude
from calcite.core.particles.particle import Particle, particle_type
from calcite.core.composites.composite import proton, neutron
from calcite.core.atoms.orbital import Orbital, orbital_type
from calcite.core.atoms.atom_functions import *
import numpy as np

# region AtomType and Atom
# region Class definitions
@structref.register
class AtomType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

orbital_key_type = types.Tuple((types.int64, types.int64, types.int64))
awi_pwi = types.Tuple((types.int64, types.int64))
awi_pwi_pwi = types.Tuple((types.int64, types.int64, types.int64))

class Atom(structref.StructRefProxy):
    def __new__(cls, protons, neutrons, electrons, ref_orbitals, orbitals, ionic_bonds, 
                covalent_bonds, position, velocity, data, index, initialized, n_electrons):

        return structref.StructRefProxy.__new__(
            cls,
            protons, neutrons, electrons,
            ref_orbitals, orbitals,
            ionic_bonds, covalent_bonds,
            position, velocity, data, index,
            initialized, n_electrons
        )
    
    @property
    def protons(self):
        return Atom_get_protons(self)
    
    @protons.setter
    def protons(self, protons):
        Atom_set_protons(self, protons)

    @property
    def neutrons(self):
        return Atom_get_neutrons(self)
    
    @neutrons.setter
    def neutrons(self, neutrons):
        Atom_set_neutrons(self, neutrons)

    @property
    def electrons(self):
        return Atom_get_electrons(self)
    
    @electrons.setter
    def electrons(self, electrons):
        Atom_set_electrons(self, electrons)
    
    @property
    def ref_orbitals(self):
        return Atom_get_ref_orbitals(self)
    
    @ref_orbitals.setter
    def ref_orbitals(self, ref_orbitals):
        Atom_set_ref_orbitals(self, ref_orbitals)

    @property
    def orbitals(self):
        return Atom_get_orbitals(self)
    
    @orbitals.setter
    def orbitals(self, orbitals):
        Atom_set_orbitals(self, orbitals)

# endregion
# region Atom functions
# region Fields
@njit
def Atom_get_protons(atom):
    return atom.protons

@njit
def Atom_set_protons(atom, protons):
    atom.protons = protons

@njit
def Atom_get_neutrons(atom):
    return atom.neutrons

@njit
def Atom_set_neutrons(atom, neutrons):
    atom.neutrons = neutrons

@njit
def Atom_get_electrons(atom):
    return atom.electrons

@njit
def Atom_set_electrons(atom, electrons):
    atom.electrons = electrons

@njit
def Atom_get_ref_orbitals(atom):
    return atom.ref_orbitals

@njit
def Atom_set_ref_orbitals(atom, ref_orbitals):
    atom.ref_orbitals = ref_orbitals

@njit
def Atom_get_orbitals(atom):
    return atom.orbitals

@njit
def Atom_set_orbitals(atom, orbitals):
    atom.orbitals = orbitals

# endregion
# endregion

structref.define_proxy(Atom, AtomType, [
    "protons", "neutrons", "electrons",
    "ref_orbitals", "orbitals",
    "ionic_bonds", "covalent_bonds",
    "position", "velocity",
    "data", "index", "initialized",
    "n_electrons"
])

#  endregion

# region Atom creation functions
@njit
def atom(n_protons, n_neutrons, n_electrons, position=None, velocity=None):
    protons = typed.List.empty_list(particle_type)
    neutrons = typed.List.empty_list(particle_type)
    electrons = typed.List.empty_list(particle_type)
    ref_orbitals = typed.Dict.empty(orbital_key_type, types.int64)
    orbitals = typed.List.empty_list(orbital_type)
    ionic_bonds = typed.List.empty_list(awi_pwi)
    covalent_bonds = typed.List.empty_list(awi_pwi_pwi)
    index = -1
    initialized = False
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    n_electrons = n_electrons
    a = Atom(
        protons, neutrons, electrons,
        ref_orbitals, orbitals,
        ionic_bonds, covalent_bonds,
        position, velocity, data, index, initialized, n_electrons
    )
    #a.init()
    return a
# endregion