import numpy as np
from dataclasses import dataclass, field
from calcite.core.particle import Particle, CompositeParticle, electron, proton, neutron, \
    particle_type, composite_particle_type
import calcite.constants as constants
from numba import njit, float64, int64, types, typed, typeof, deferred_type
from numba.experimental import jitclass
from numba.types import Tuple


orbital_spec = [
    ('n', int64),
    ('l', int64),
    ('m', int64),
    ('electrons', types.ListType(particle_type))
]

@jitclass(orbital_spec)
class Orbital:
    def __init__(self, n, l, m, electrons):
        self.n = n
        self.l = l
        self.m = m
        self.electrons = electrons

    def add(self, spin: float):
        if len(self.electrons) < 2 and spin not in [_electron.spin for _electron in self.electrons]:
            _electron = electron(self.n, self.l, self.m)
            _electron.spin = spin
            self.electrons.append(_electron)
            return True
        return False

orbital_type = typeof(Orbital(1, 0, 0, typed.List.empty_list(particle_type)))
orbital_key_type = Tuple((int64, int64, int64))

_atom_type = deferred_type()
atom_spec = [
    ('position', float64[:]),
    ('velocity', float64[:]),
    ('protons', types.ListType(composite_particle_type)),
    ('neutrons', types.ListType(composite_particle_type)),
    ('electrons', types.ListType(particle_type)),
    ('orbitals', types.DictType(orbital_key_type, int64)),
    ('_orbitals', types.ListType(orbital_type)),
    ('ionic_bonds', types.ListType(types.Tuple([_atom_type, particle_type]))),
    ('covalent_bonds', types.ListType(types.Tuple([_atom_type, particle_type]))),
]

@jitclass(atom_spec)
class Atom:
    def __init__(self, position, velocity, n_protons, n_neutrons, n_electrons):
        self.position = position
        self.velocity = velocity
        self.protons = typed.List([proton() for _ in range(n_protons)])
        self.neutrons = typed.List([neutron() for _ in range(n_neutrons)])
        self.electrons = typed.List.empty_list(particle_type)
        self.orbitals = typed.Dict.empty(orbital_key_type, int64)
        self._orbitals = typed.List.empty_list(orbital_type)
        self.configure(n_electrons)

    def configure(self, n_electrons: int):
        order = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0)]
        added = 0
        states = [0.5, -0.5]
        for n, l in order:
            for m in range(-l, l + 1):
                if (n, l, m) not in self.orbitals:
                    self.orbitals[(n, l, m)] = len(self._orbitals)
                    self._orbitals.append(Orbital(n, l, m, typed.List.empty_list(particle_type)))
                orbital = self._orbitals[self.orbitals[(n, l, m)]]
                for spin in states:
                    if added < n_electrons and orbital.add(spin):
                        self.electrons.append(orbital.electrons[-1])
                        added += 1
            if added >= n_electrons:
                return
            
    def add_electron(self, _electron: Particle):
        for orbital in self._orbitals:
            if orbital.add(_electron.spin):
                self.electrons.append(_electron)
                return True
        return False
    
    def remove_electron(self):
        if self.electrons:
            _electron = self.electrons.pop()
            orbital = self.orbitals[(_electron["n"], _electron["l"], _electron["m"])]
            orbital.electrons.remove(_electron)
            return _electron
        return None
    
    def covalent_bond(self, atom: "Atom"):
        for orbital in self._orbitals:
            if len(orbital.electrons) == 1:
                for other in atom._orbitals:
                    if len(other.electrons) == 1:
                        orbital.electrons.append(other.electrons[0])
                        other.electrons.append(orbital.electrons[0])
                        self.covalent_bonds.append((atom, orbital.electrons[0]))
                        atom.covalent_bonds.append((self, other.electrons[0]))
                        return True
        return False

    def ionic_bond(self, atom: "Atom"):
        if self.charge() <= 0 and atom.charge() >= 0:
            _electron = self.remove_electron()
            if _electron:
                atom.add_electron(_electron)
                self.ionic_bonds.append((atom, _electron))
                atom.ionic_bonds.append((self, _electron))
                return True
        return False

    @property
    def mass(self):
        return sum([proton.mass for proton in self.protons]) \
            + sum([neutron.mass for neutron in self.neutrons]) \
            + sum([_electron.mass for _electron in self.electrons])
    
    @property
    def atomic_number(self):
        return len(self.protons)
    
    @property
    def charge(self):
        return sum([proton.charge for proton in self.protons]) \
            - sum([_electron.charge for _electron in self.electrons])
    
    @property
    def spin(self):
        return sum([proton.spin for proton in self.protons]) \
            + sum([neutron.spin for neutron in self.neutrons]) \
            + sum([_electron.spin for _electron in self.electrons])
    
    @property
    def momentum(self):
        return np.linalg.norm(self.mass * self.velocity)

@njit
def atom(n_protons: int, n_neutrons: int, n_electrons: int, position: list[float] | None = None, 
         velocity: list[float] | None = None) -> Atom:
    if position is None:
        position = constants.NAN_VECTOR
    if velocity is None:
        velocity = constants.NAN_VECTOR
    return Atom(position, velocity, n_protons, n_neutrons, n_electrons)

atom_type = typeof(atom(0, 0, 0))
_atom_type.define(atom_type)