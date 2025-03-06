from numba.experimental import structref
from numba import njit, types, typed
from numba.extending import overload_method
from calcite.formulas import magnitude
from calcite.core.particles.particle import Particle, ParticleType
from calcite.core.composites.composite import proton, neutron
from calcite.core.atoms.orbital import Orbital, OrbitalType
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
    
        