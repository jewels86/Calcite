import numpy as np
from dataclasses import dataclass, field
from calcite.core.particle import Particle, CompositeParticle, electron, proton, neutron, \
    particle_type, composite_particle_type
from calcite.constants import NAN_VECTOR
from calcite.core.exceptions import UnassignedElectronsError
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
awi_electron_electron = types.Tuple([int64, particle_type, particle_type])
atom_spec = [
    ('position', float64[:]), # position vector (x, y, z)
    ('velocity', float64[:]), # velocity vector (vx, vy, vz)
    ('protons', types.ListType(composite_particle_type)), # list of protons
    ('neutrons', types.ListType(composite_particle_type)), # list of neutrons
    ('electrons', types.ListType(particle_type)), # list of electrons
    ('orbitals', types.DictType(orbital_key_type, int64)), # dictionary of orbital parameters -> index
    ('_orbitals', types.ListType(orbital_type)), # list of orbitals
    ('ionic_bonds', types.ListType(types.Tuple([int64, particle_type]))), # list of ionic bonds (atom world index -> electron)
    ('covalent_bonds', types.ListType(awi_electron_electron)), # list of covalent bonds (atom world index -> electron)
    ('unassigned', types.ListType(particle_type)), # list of unassigned electrons
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
        self.covalent_bonds = typed.List.empty_list(awi_electron_electron)
        self.unassigned = typed.List.empty_list(particle_type)
        self.index = -1
        self.setup_electrons(n_electrons)
        self.configure()

    def setup_electrons(self, n_electrons):
        added = 0
        orbitals_order = [(n, l) for n in range(1, 7) for l in range(n)]
        
        for n, l in orbitals_order:
            for m in range(-l, l+1):
                if added < n_electrons:
                    new_electron = electron(n, l, m)
                    new_electron.spin = 0.5 if added % 2 == 0 else -0.5
                    self.electrons.append(new_electron)
                    added += 1
        
    def configure(self, order: list = None) -> None:
        added = 0
        if order is None: order = [(n, l) for n in range(1, 7) for l in range(n)]
        capacity = sum([2 * (2 * l + 1) for _, l in order])
        unassigned = typed.List.empty_list(particle_type)

        for orbital in self._orbitals:
            orbital.electrons.clear()

        for n, l in order:
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
                            unassigned.append(electron)
                    else:
                        break
        
        to_stay = [orbital for orbital in self._orbitals if len(orbital.electrons) != 0]
        to_remove = [orbital for orbital in self._orbitals if len(orbital.electrons) == 0]
        for orbital in to_remove: self.orbitals.pop((orbital.n, orbital.l, orbital.m), None)
        self._orbitals = typed.List(to_stay)

        if added < len(self.electrons):
            print(f"Warning in calcite.Atom.configure: {len(self.electrons) - added} electrons remain unassigned.")
            print(f"Unassigned electrons will be stored in the unassigned list.")
            raise UnassignedElectronsError("Unassigned electrons remain.")
        
        if len(self.electrons) > capacity:
            print(f"Warning in calcite.Atom.configure: Atom has {len(self.electrons)} electrons, but only {capacity} can be assigned.")
            print(f"Excess electrons will be stored in the unassigned list.")
            raise UnassignedElectronsError("Excess electrons remain.")
        self.unassigned = unassigned
    
    def add_electron(self, electron: Particle) -> bool:
        for orbital in self._orbitals:
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
    
    def remove_electron(self, electron: Particle) -> bool:
        new_electrons = typed.List.empty_list(particle_type)
        removed = False

        for e in self.electrons:
            if e != electron or removed:
                new_electrons.append(e)
            else:
                removed = True

        if removed:
            self.electrons = new_electrons
            self.configure()
            return True
        return False
    
    def covalent_bond(self, other: "Atom") -> bool:
        if self.stable or other.stable:
            return False
        
        valence_self = self.valence_electrons
        valence_other = other.valence_electrons

        unpaired_self = [e for e in valence_self if e.spin == 0.5]
        unpaired_other = [e for e in valence_other if e.spin == 0.5]

        if len(unpaired_self) == 0 or len(unpaired_other) == 0:
            return False
        
        self.covalent_bonds.append((other.index, unpaired_self[0], unpaired_other[0]))
        other.covalent_bonds.append((self.index, unpaired_other[0], unpaired_self[0]))

        unpaired_self[0].spin = -0.5
        unpaired_other[0].spin = -0.5

        self.remove_electron(unpaired_self[0])
        other.remove_electron(unpaired_other[0])

        self.add_electron(unpaired_other[0])
        other.add_electron(unpaired_self[0])

        return True
    
    def ionic_bond(self, other: "Atom") -> bool:
        if self.stable or other.stable:
            return False
        
        unpaired_self = [e for e in self.valence_electrons if e.spin == 0.5]
        unpaired_other = [e for e in other.valence_electrons if e.spin == 0.5]

        if len(unpaired_self) > 0 and len(unpaired_other) > 0:
            e_to_transfer = unpaired_self[0]
            self.electrons.remove(e_to_transfer)
            other.electrons.append(e_to_transfer)
            e_to_transfer.spin = -0.5
            self.ionic_bonds.append((other.index, e_to_transfer))
            other.ionic_bonds.append((self.index, e_to_transfer))

            return True
        return False

    @property
    def valence_electrons(self):
        valence_electrons = []

        for orbital in self._orbitals:
            if orbital.n == max([o.n for o in self._orbitals]):
                valence_electrons.extend(orbital.electrons)

        return valence_electrons

    @property
    def stable(self):
        valence_electrons = 0
        max_valence = 8 if self.atomic_number < 18 else 18

        for orbital in self._orbitals:
            if orbital.n == max([o.n for o in self._orbitals]):
                valence_electrons += len(orbital.electrons)
        
        return valence_electrons == max_valence

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
        position = NAN_VECTOR
    if velocity is None:
        velocity = NAN_VECTOR
    return Atom(position, velocity, n_protons, n_neutrons, n_electrons)

atom_type = typeof(atom(0, 0, 0))