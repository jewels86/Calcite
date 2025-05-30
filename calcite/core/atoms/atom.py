from numba.experimental import structref
from numba import njit, types, typed, int64
from numba.extending import overload_method
from calcite.formulas import orbital_order
from calcite.core.particles.particle import Particle, particle_type, electron
from calcite.core.composites.composite import proton, neutron, composite_particle_type
from calcite.core.atoms.orbital import orbital, orbital_type
from calcite.core.atoms.atom_functions import *
from calcite.core.vectors.vector import vector, vector_type
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
                covalent_bonds, position, velocity, data, index, debug_mode, n_electrons, symbol):
        if isinstance(position, (list, np.ndarray)):
            position = vector(position)
        if isinstance(velocity, (list, np.ndarray)):
            velocity = vector(velocity)
        return structref.StructRefProxy.__new__(
            cls,
            protons, neutrons, electrons,
            ref_orbitals, orbitals,
            ionic_bonds, covalent_bonds,
            position, velocity, data, index,
            debug_mode, n_electrons, symbol
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
        return Atom_get_log(self)
    
    @_debug.setter
    def _debug(self, log):
        Atom_set_log(self, log)

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
    def symbol(self):
        return Atom_get_symbol(self)
    
    @symbol.setter
    def symbol(self, symbol):
        Atom_set_symbol(self, symbol)

# endregion
# region Atom functions
# region Fields
@njit(cache=True)
def Atom_get_protons(atom):
    return atom.protons

@njit(cache=True)
def Atom_set_protons(atom, protons):
    atom.protons = protons

@njit(cache=True)
def Atom_get_neutrons(atom):
    return atom.neutrons

@njit(cache=True)
def Atom_set_neutrons(atom, neutrons):
    atom.neutrons = neutrons

@njit(cache=True)
def Atom_get_electrons(atom):
    return atom.electrons

@njit(cache=True)
def Atom_set_electrons(atom, electrons):
    atom.electrons = electrons

@njit(cache=True)
def Atom_get_ref_orbitals(atom):
    return atom.ref_orbitals

@njit(cache=True)
def Atom_set_ref_orbitals(atom, ref_orbitals):
    atom.ref_orbitals = ref_orbitals

@njit(cache=True)
def Atom_get_orbitals(atom):
    return atom.orbitals

@njit(cache=True)
def Atom_set_orbitals(atom, orbitals):
    atom.orbitals = orbitals

@njit(cache=True)
def Atom_get_ionic_bonds(atom):
    return atom.ionic_bonds

@njit(cache=True)
def Atom_set_ionic_bonds(atom, ionic_bonds):
    atom.ionic_bonds = ionic_bonds

@njit(cache=True)
def Atom_get_covalent_bonds(atom):
    return atom.covalent_bonds

@njit(cache=True)
def Atom_set_covalent_bonds(atom, covalent_bonds):
    atom.covalent_bonds = covalent_bonds

@njit(cache=True)
def Atom_get_position(atom):
    return atom.position

@njit(cache=True)
def Atom_set_position(atom, position):
    atom.position = position

@njit(cache=True)
def Atom_get_velocity(atom):
    return atom.velocity

@njit(cache=True)
def Atom_set_velocity(atom, velocity):
    atom.velocity = velocity

@njit(cache=True)
def Atom_get_data(atom):
    return atom.data

@njit(cache=True)
def Atom_set_data(atom, data):
    atom.data = data

@njit(cache=True)
def Atom_get_index(atom):
    return atom.index

@njit(cache=True)
def Atom_set_index(atom, index):
    atom.index = index

@njit(cache=True)
def Atom_get_debug_mode(atom):
    return atom.debug_mode

@njit(cache=True)
def Atom_set_debug_mode(atom, debug_mode):
    atom.debug_mode = debug_mode

@njit(cache=True)
def Atom_get_n_electrons(atom):
    return atom.n_electrons

@njit(cache=True)
def Atom_set_n_electrons(atom, n_electrons):
    atom.n_electrons = n_electrons

@njit(cache=True)
def Atom_get_log(atom):
    return atom._debug

@njit(cache=True)
def Atom_set_log(atom, log):
    atom._debug = log

@njit(cache=True)
def Atom_get_symbol(atom):
    return atom.symbol

@njit(cache=True)
def Atom_set_symbol(atom, symbol):
    atom.symbol = symbol

# endregion
# region Methods
@overload_method(AtomType, "configure")
def Atom_configure(self):
    def impl(self):
        return configure(self)
    return impl

@overload_method(AtomType, "add")
def Atom_add(self, electron):
    def impl(self, electron):
        return add(self, electron)
    return impl

@overload_method(AtomType, "remove")
def Atom_remove(self):
    def impl(self):
        return remove(self)
    return impl

@overload_method(AtomType, "remove_specific")
def Atom_remove_specific(self, electron):
    def impl(self, electron):
        return remove_specific(self, electron)
    return impl

@overload_method(AtomType, "valence_electrons")
def Atom_valence_electrons(self):
    def impl(self):
        return valence_electrons(self)
    return impl

@overload_method(AtomType, "stable")
def Atom_stable(self):
    def impl(self):
        return stable(self)
    return impl

@overload_method(AtomType, "add_to_valence_shell")
def Atom_add_to_valence_shell(self, electron):
    def impl(self, electron):
        return add_to_valence_shell(self, electron)
    return impl

@overload_method(AtomType, "remove_from_valence_shell")
def Atom_remove_from_valence_shell(self):
    def impl(self):
        return remove_from_valence_shell(self)
    return impl

@overload_method(AtomType, "covalent_bond")
def Atom_covalent_bond(self, other):
    def impl(self, other):
        return covalent_bond(self, other)
    return impl

@overload_method(AtomType, "ionic_bond")
def Atom_ionic_bond(self, other):
    def impl(self, other):
        return ionic_bond(self, other)
    return impl

@overload_method(AtomType, "kinetic_energy")
def Atom_kinetic_energy(self):
    def impl(self):
        return 0.5 * self.mass * self.velocity.magnitude() ** 2
    return impl

@overload_method(AtomType, "momentum")
def Atom_momentum(self):
    def impl(self):
        return self.mass * self.velocity
    return impl

@overload_method(AtomType, "charge")
def Atom_charge(self):
    def impl(self):
        charge = 0.0
        for e in self.electrons:
            charge += e.charge
        for p in self.protons:
            charge += p.charge()
        return charge
    return impl

@overload_method(AtomType, "mass")
def Atom_mass(self):
    def impl(self):
        mass = 0.0
        for p in self.protons:
            mass += p.mass()
        for n in self.neutrons:
            mass += n.mass()
        for e in self.electrons:
            mass += e.mass
        return mass
    return impl

# endregion
# endregion

structref.define_proxy(Atom, AtomType, [
    "protons", "neutrons", "electrons",
    "ref_orbitals", "orbitals",
    "ionic_bonds", "covalent_bonds",
    "position", "velocity",
    "data", "index", "debug_mode",
    "n_electrons", "symbol"
])

atom_type = AtomType(
    fields=[
        ("protons", types.ListType(composite_particle_type)),
        ("neutrons", types.ListType(composite_particle_type)),
        ("electrons", types.ListType(particle_type)),
        ("ref_orbitals", types.DictType(orbital_key_type, types.int64)),
        ("orbitals", types.ListType(orbital_type)),
        ("ionic_bonds", types.ListType(awi_pwi)),
        ("covalent_bonds", types.ListType(awi_pwi_pwi)),
        ("position", vector_type),
        ("velocity", vector_type),
        ("data", types.DictType(types.unicode_type, types.float64)),
        ("index", int64),
        ("debug_mode", types.boolean),
        ("n_electrons", int64),
        ("symbol", types.unicode_type),
    ]
)

#  endregion

# region Atom creation functions
@njit(cache=True)
def atom(n_protons, n_neutrons, n_electrons, symbol="UNNAMED", position=None, velocity=None, debug_mode=False):
    protons = typed.List([proton() for _ in range(n_protons)])
    neutrons = typed.List([neutron() for _ in range(n_neutrons)])
    electrons = typed.List.empty_list(particle_type)
    ref_orbitals = typed.Dict.empty(orbital_key_type, types.int64)
    orbitals = typed.List.empty_list(orbital_type)
    ionic_bonds = typed.List.empty_list(awi_pwi)
    covalent_bonds = typed.List.empty_list(awi_pwi_pwi)
    index = -1
    data = typed.Dict.empty(types.unicode_type, types.float64)
    n_electrons = n_electrons
    if debug_mode: print(f"Creating atom with {n_protons} protons, {n_neutrons} neutrons, and {n_electrons} electrons.")
    
    position = vector(*position) if position is not None else vector(np.nan, np.nan, np.nan)
    velocity = vector(*velocity) if velocity is not None else vector(np.nan, np.nan, np.nan)

    a = Atom(
        protons, neutrons, electrons,
        ref_orbitals, orbitals,
        ionic_bonds, covalent_bonds,
        position, velocity, data, 
        index, debug_mode, n_electrons, symbol
    )
    
    a.configure()

    return a
# endregion