from numba.experimental import structref
from numba import njit, types, typed, int64
from numba.extending import overload_method
from calcite.formulas import orbital_order
from calcite.core.particles.particle import Particle, particle_type, electron
from calcite.core.composites.composite import proton, neutron
from calcite.core.atoms.orbital import orbital, orbital_type
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
                covalent_bonds, position, velocity, data, index, debug_mode, n_electrons, _debug):

        return structref.StructRefProxy.__new__(
            cls,
            protons, neutrons, electrons,
            ref_orbitals, orbitals,
            ionic_bonds, covalent_bonds,
            position, velocity, data, index,
            debug_mode, n_electrons, _debug
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

    @property
    def ionic_bonds(self):
        return Atom_get_ionic_bonds(self)
    
    @ionic_bonds.setter
    def ionic_bonds(self, ionic_bonds):
        Atom_set_ionic_bonds(self, ionic_bonds)

    @property
    def covalent_bonds(self):
        return Atom_get_covalent_bonds(self)
    
    @covalent_bonds.setter
    def covalent_bonds(self, covalent_bonds):
        Atom_set_covalent_bonds(self, covalent_bonds)

    @property
    def position(self):
        return Atom_get_position(self)
    
    @position.setter
    def position(self, position):
        Atom_set_position(self, position)

    @property
    def velocity(self):
        return Atom_get_velocity(self)
    
    @velocity.setter
    def velocity(self, velocity):
        Atom_set_velocity(self, velocity)

    @property
    def data(self):
        return Atom_get_data(self)
    
    @data.setter
    def data(self, data):
        Atom_set_data(self, data)

    @property
    def index(self):
        return Atom_get_index(self)
    
    @index.setter
    def index(self, index):
        Atom_set_index(self, index)

    @property
    def _debug(self):
        return Atom_get_debug_mode(self)
    
    @_debug.setter
    def _debug(self, debug_mode):
        Atom_set_debug_mode(self, debug_mode)

    @property
    def debug_mode(self):
        return Atom_get_debug_mode(self)
    
    @debug_mode.setter
    def debug_mode(self, debug_mode):
        Atom_set_debug_mode(self, debug_mode)

    @property
    def n_electrons(self):
        return Atom_get_n_electrons(self)
    
    @n_electrons.setter
    def n_electrons(self, n_electrons):
        Atom_set_n_electrons(self, n_electrons)

    @property
    def _debug(self):
        return Atom_get_debug_function(self)
    
    @_debug.setter
    def _debug(self, _debug):
        Atom_set_debug_function(self, _debug)

    @property
    def debug_function(self):
        return Atom_get_debug_function(self)
    
    @debug_function.setter
    def debug_function(self, _debug):
        Atom_set_debug_function(self, _debug)

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

@njit
def Atom_get_ionic_bonds(atom):
    return atom.ionic_bonds

@njit
def Atom_set_ionic_bonds(atom, ionic_bonds):
    atom.ionic_bonds = ionic_bonds

@njit
def Atom_get_covalent_bonds(atom):
    return atom.covalent_bonds

@njit
def Atom_set_covalent_bonds(atom, covalent_bonds):
    atom.covalent_bonds = covalent_bonds

@njit
def Atom_get_position(atom):
    return atom.position

@njit
def Atom_set_position(atom, position):
    atom.position = position

@njit
def Atom_get_velocity(atom):
    return atom.velocity

@njit
def Atom_set_velocity(atom, velocity):
    atom.velocity = velocity

@njit
def Atom_get_data(atom):
    return atom.data

@njit
def Atom_set_data(atom, data):
    atom.data = data

@njit
def Atom_get_index(atom):
    return atom.index

@njit
def Atom_set_index(atom, index):
    atom.index = index

@njit
def Atom_get_debug_mode(atom):
    return atom.debug_mode

@njit
def Atom_set_debug_mode(atom, debug_mode):
    atom.debug_mode = debug_mode

@njit
def Atom_get_n_electrons(atom):
    return atom.n_electrons

@njit
def Atom_set_n_electrons(atom, n_electrons):
    atom.n_electrons = n_electrons

@njit
def Atom_get_debug_function(atom):
    return atom._debug

@njit
def Atom_set_debug_function(atom, _debug):
    atom._debug = _debug

# endregion
# endregion

structref.define_proxy(Atom, AtomType, [
    "protons", "neutrons", "electrons",
    "ref_orbitals", "orbitals",
    "ionic_bonds", "covalent_bonds",
    "position", "velocity",
    "data", "index", "debug_mode",
    "n_electrons", "_debug"
])

#  endregion

# region Atom creation functions
@njit
def atom(n_protons, n_neutrons, n_electrons, position=None, velocity=None, debug_mode=False, _debug=None):
    protons = typed.List([proton() for _ in range(n_protons)])
    neutrons = typed.List([neutron() for _ in range(n_neutrons)])
    electrons = typed.List.empty_list(particle_type)
    ref_orbitals = typed.Dict.empty(orbital_key_type, types.int64)
    orbitals = typed.List.empty_list(orbital_type)
    ionic_bonds = typed.List.empty_list(awi_pwi)
    covalent_bonds = typed.List.empty_list(awi_pwi_pwi)
    index = -1
    data = typed.Dict.empty(types.unicode_type, types.unicode_type)
    n_electrons = n_electrons
    if _debug is None:
        def func(location, severity, content):
            print(f"[{location} - {severity}]: {content}")
        _debug = func
    a = Atom(
        protons, neutrons, electrons,
        ref_orbitals, orbitals,
        ionic_bonds, covalent_bonds,
        position, velocity, data, index, debug_mode, n_electrons, _debug
    )
    
    added = 0
    order = orbital_order(n_electrons)

    for n, l in order:
        for m in range(-l, l+1):
            if added == n_electrons:
                break
            electrons = []
            new_electron = electron(n, l, m, 0.5 if added % 2 == 0 else -0.5)
            electrons.append(new_electron)
    
    a.configure()

    return a
# endregion