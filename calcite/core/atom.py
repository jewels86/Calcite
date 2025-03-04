import numpy as np
from dataclasses import dataclass, field
from calcite.core.particle import Particle, CompositeParticle, electron, proton, neutron, \
    particle_type, composite_particle_type
from calcite import constants
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
        self.n = n # principal quantum number
        self.l = l # azimuthal quantum number
        self.m = m # magnetic quantum number
        self.electrons = electrons

    def can_add(self, electron: Particle):
        return len(self.electrons) < 2 and electron.spin not in [e.spin for e in self.electrons]
        # orbitals can only hold two electrons - one with spin up and one with spin down

    def add(self, electron: Particle):
        if self.can_add(electron): 
            self.electrons.append(electron) 
            return True
        return False
        

orbital_type = typeof(Orbital(1, 0, 0, typed.List.empty_list(particle_type)))
orbital_key_type = Tuple((int64, int64, int64))
awi_electron = types.Tuple([int64, particle_type])

atom_spec = [
    ('position', float64[:]), # position vector (x, y, z)
    ('velocity', float64[:]), # velocity vector (vx, vy, vz)
    ('protons', types.ListType(composite_particle_type)), # list of protons
    ('neutrons', types.ListType(composite_particle_type)), # list of neutrons
    ('electrons', types.ListType(particle_type)), # list of electrons
    ('orbitals', types.DictType(orbital_key_type, int64)), # dictionary of orbital parameters -> index
    ('_orbitals', types.ListType(orbital_type)), # list of orbitals
    ('ionic_bonds', types.ListType(types.Tuple([int64, particle_type]))), # list of ionic bonds (atom world index -> electron)
    ('covalent_bonds', types.ListType(types.Tuple([int64, particle_type]))), # list of covalent bonds (atom world index -> electron)
    ('index', int64) # atom world index
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
        self.ionic_bonds = typed.List.empty_list(awi_electron)
        self.covalent_bonds = typed.List.empty_list(awi_electron)
        
    def configure(self, order: list = constants.ORBITAL_ORDER) -> None:
        added = 0

        for orbital in self._orbitals:
            orbital.electrons = typed.List.empty_list(particle_type)

        for n, l in order:
            orbitals = []
            for m in range(-l, l+1):
                if (n, l, m) not in self.orbitals:
                    self.orbitals[(n, l, m)] = len(self._orbitals)
                    self._orbitals.append(Orbital(n, l, m, typed.List.empty_list(particle_type)))
                orbital = self._orbitals[self.orbitals[(n, l, m)]]
                for _ in range(2):
                    if added < len(self.electrons):
                        electron = self.electrons[added]
                        if orbital.add(electron):
                            added += 1
                    else:
                        break

        for orbital in self._orbitals:
            if len(orbital.electrons) == 0:
                self.orbitals.pop((orbital.n, orbital.l, orbital.m))
                self._orbitals.remove(orbital)
    
    def add_electron(self, electron: Particle) -> bool:
        for orbital in self._orbtials:
            if orbital.can_add(electron):
                self.electrons.append(electron)
                self.configure()
                return True
        return False
    
    def remove_electron(self) -> bool:
        if len(self.electrons) > 0:
            self.electrons.pop()
            self.configure()
            return True
        return False
    
    def covalent_bond(self, other: "Atom") -> bool:
        

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