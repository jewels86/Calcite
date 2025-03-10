from numba.experimental import structref
from numba import njit, types, typed
from numba.extending import overload_method
from calcite.formulas import magnitude
from calcite.core.particles.particle import Particle, ParticleType
from calcite.core.composites.composite import proton, neutron
from calcite.core.atoms.orbital import Orbital, OrbitalType
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
    def __new__(cls, n_protons, n_neutrons, n_electrons, position, velocity, data):
        protons = typed.List([proton() for _ in range(n_protons)])
        neutrons = typed.List([neutron() for _ in range(n_neutrons)])
        electrons = typed.List.empty_list(ParticleType)
        orbitals = typed.Dict.empty(orbital_key_type, types.int64)
        _orbitals = typed.List.empty_list(OrbitalType)
        ionic_bonds = typed.List.empty_list(awi_pwi)
        covalent_bonds = typed.List.empty_list(awi_pwi_pwi)
        index = -1
        initialized = False

        return structref.StructRefProxy.__new__(
            cls,
            protons, neutrons, electrons,
            orbitals, _orbitals,
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
    def orbitals(self):
        return Atom_get_orbitals(self)
    
    @orbitals.setter
    def orbitals(self, orbitals):
        Atom_set_orbitals(self, orbitals)

    @property
    def _orbitals(self):
        return Atom_get__orbitals(self)
    
    @_orbitals.setter
    def _orbitals(self, _orbitals):
        Atom_set__orbitals(self, _orbitals)

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
def Atom_get_orbitals(atom):
    return atom.orbitals

@njit
def Atom_set_orbitals(atom, orbitals):
    atom.orbitals = orbitals

@njit
def Atom_get__orbitals(atom):
    return atom._orbitals

@njit
def Atom_set__orbitals(atom, _orbitals):
    atom._orbitals = _orbitals