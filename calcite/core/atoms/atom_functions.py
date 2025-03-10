import numpy as np
from numba import njit, typed, types
from calcite.core.composites.composite import proton, neutron 
from calcite.core.particles.particle import electron, ParticleType, Particle
from calcite.core.atoms.orbital import Orbital, OrbitalType

@njit
def init(self, max_n=-1):
    added = 0
    n_electrons = self.n_electrons
    max_n = int(np.ceil((n_electrons / 2)**(1/3))) if max_n == -1 else max_n
    orbitals_order = [(n, l) for n in range(1, max_n) for l in range(n)]
    if self.debug_mode:
        print(f"Atom.init: Initializing atom with {n_electrons} electrons (max_n={max_n}, orbitals_order_len={len(orbitals_order)})")

    for n, l in orbitals_order:
            for m in range(-l, l+1):
                if added < n_electrons:
                    new_electron = electron(n, l, m)
                    new_electron.spin = 0.5 if added % 2 == 0 else -0.5
                    self.electrons.append(new_electron)
                    if self.debug_mode: print(f"Added electron with spin {'up' if new_electron.spin == 0.5 else 'down'} to orbital {n}, {l}, {m}")
                    added += 1

    if added < n_electrons:
        if self.debug_mode:
            print(f"Atom.init: {max_n} max_n was undercalculated or underset, remaining electrons: {n_electrons - added}")
            print(f"Atom.init: Adding remaining electrons to higher orbitals (estimated number of orbitals: {int(np.ceil((n_electrons - added) / 2))})")
        for n in range(max_n, int(np.ceil((n_electrons - added) / 2))):
            for l in range(n):
                for m in range(-l, l+1):
                    if added < n_electrons:
                        new_electron = electron(n, l, m)
                        new_electron.spin = 0.5 if added % 2 == 0 else -0.5
                        self.electrons.append(new_electron)
                        if self.debug_mode: print(f"Added electron with spin {'up' if new_electron.spin == 0.5 else 'down'} to orbital {n}, {l}, {m}")
                        added += 1
    if self.debug_mode: print(f"Atom.init: Initialized atom with {n_electrons} electrons")

@njit(cache=True)
def configure(self):
    added = 0
    max_n = int(np.ceil((self.n_electrons / 2)**(1/3)))
    order = [(n, l) for n in range(1, max_n) for l in range(n)]
    if self.debug_mode:
        print(f"Atom.configure: Configuring atom with {self.n_electrons} electrons (max_n={max_n}, orbitals_order_len={len(order)})")
    
    capacity = sum([2 * (2 * l + 1) for _, l in order])
    unassigned = typed.List.empty_list(ParticleType)
    if capacity < len(self.electrons):
        print(f"Atom.configure: Atom has {len(self.electrons)} electrons, but only {capacity} can be assigned.")
        print(f"Atom.configure: order will be reevalulated")
        max_n = int(np.ceil((len(self.electrons) / 2)**(1/3))) + 1
        order = [(n, l) for n in range(1, max_n) for l in range(n)]
        capacity = sum([2 * (2 * l + 1) for _, l in order])
        print(f"Atom.configure: New max_n: {max_n}, new capacity: {capacity}")
    

    for orbital in self._orbitals:
        orbital.electrons.clear()

    for n, l in order:
        for m in range(-l, l+1):
            if (n, l, m) not in self.orbitals:
                self.orbitals[(n, l, m)] = len(self._orbitals)
                self._orbitals.append(Orbital(n, l, m, typed.List.empty_list(ParticleType)))
            orbital = self._orbitals[self.orbitals[(n, l, m)]]
            for _ in range(2):
                if added < len(self.electrons):
                    electron = self.electrons[added]
                    if orbital.add(electron):
                        if self.debug_mode: 
                            print(f"Assigned electron with spin {'up' if electron.spin == 0.5 else 'down'} to orbital {n}, {l}, {m}")
                        added += 1
                    else:
                        unassigned.append(electron)
                else:
                    break

    max_n = max([orbital.n for orbital in self._orbitals])
    for n in range(1, max_n + 2):
        for l in range(n):
            for m in range(-l, l + 1):
                if (n, l, m) not in self.orbitals:
                    self.orbitals[(n, l, m)] = len(self._orbitals)
                    self._orbitals.append(Orbital(n, l, m, typed.List.empty_list(ParticleType)))

    to_stay = [orbital for orbital in self._orbitals if len(orbital.electrons) != 0]
    to_remove = [orbital for orbital in self._orbitals if len(orbital.electrons) == 0]
    for orbital in to_remove: self.orbitals.pop((orbital.n, orbital.l, orbital.m), None)
    self._orbitals = typed.List(to_stay)

@njit
def add_electron(self, electron: Particle):
    for orbital in self._orbitals:
        if orbital.can_add(electron):
            self.electrons.append(electron)
            self.configure()
            return True
    return False

@njit
def remove_electron(self):
    if len(self.electrons) > 0:
        self.electrons.pop()
        self.configure()
        return True
    return False

@njit 
def remove_specifc_electron(self, electron: Particle):
    if electron in self.electrons:
        self.electrons.remove(electron)
        self.configure()
        return True
    return False

def covalent_bond(self, other) -> bool:
    if self.stable or other.stable:
        return False
    
    valence_self = self.valence_electrons
    valence_other = other.valence_electrons

    unpaired_self = [e for e in valence_self if e.spin == 0.5]
    unpaired_other = [e for e in valence_other if e.spin == 0.5]

    if len(unpaired_self) == 0 or len(unpaired_other) == 0:
        return False

    original_self_spin = unpaired_self[0].spin
    original_other_spin = unpaired_other[0].spin
    unpaired_self[0].spin = -0.5
    unpaired_other[0].spin = 0.5

    if not self.add_electron_to_valence_shell(unpaired_other[0]) or \
        not other.add_electron_to_valence_shell(unpaired_self[0]):
        unpaired_self[0].spin = 0.5
        unpaired_other[0].spin = -0.5
        if not self.add_electron_to_valence_shell(unpaired_other[0]) or \
            not other.add_electron_to_valence_shell(unpaired_self[0]):
            unpaired_self[0].spin = original_self_spin
            unpaired_other[0].spin = original_other_spin
            return False
    
    self.covalent_bonds.append((other.index, unpaired_self[0], unpaired_other[0]))
    other.covalent_bonds.append((self.index, unpaired_other[0], unpaired_self[0]))

    return True

def add_electron_to_valence_shell(self, electron) -> bool:
    valence_shell = max([orbital.n for orbital in self._orbitals])
    if self.debug_mode: print(f"Adding electron to valence shell {valence_shell}")
    
    for orbital in self._orbitals:
        if orbital.n == valence_shell and orbital.can_add(electron):
            if self.debug_mode: print(f"Adding electron to orbital {orbital.n}, {orbital.l}, {orbital.m}")
            orbital.add(electron)
            self.electrons.append(electron)
            return True
    
    for l in range(valence_shell):
        for m in range(-l, l + 1):
            if (valence_shell, l, m) not in self.orbitals:
                if self.debug_mode: print(f"Creating new orbital {valence_shell}, {l}, {m} for electron")
                self.orbitals[(valence_shell, l, m)] = len(self._orbitals)
                new_orbital = Orbital(valence_shell, l, m, typed.List.empty_list(ParticleType))
                self._orbitals.append(new_orbital)
                if new_orbital.can_add(electron):
                    if self.debug_mode: print(f"Adding electron to new orbital {valence_shell}, {l}, {m}")
                    new_orbital.add(electron)
                    self.electrons.append(electron)
                    return True
    
    if self.debug_mode: print("Failed to add electron to valence shell")
    return False

@njit
def valence_electrons(self):
    for orbital in self._orbitals:
        if orbital.n == max([o.n for o in self._orbitals]):
            valence_electrons.extend(orbital.electrons)

    return valence_electrons

@njit
def stable(self):
    valence_electrons = 0
    max_valence = 8 if self.atomic_number < 18 else 18

    for orbital in self._orbitals:
        if orbital.n == max([o.n for o in self._orbitals]):
            valence_electrons += len(orbital.electrons)
    
    return valence_electrons == max_valence or valence_electrons == 0 or valence_electrons == 2

@njit
def mass(self):
    return sum([proton.mass for proton in self.protons]) \
        + sum([neutron.mass for neutron in self.neutrons]) \
        + sum([_electron.mass for _electron in self.electrons])

@njit
def atomic_number(self):
    return len(self.protons)

@njit
def charge(self):
    return sum([proton.charge for proton in self.protons]) \
        + sum([_electron.charge for _electron in self.electrons])

@njit
def spin(self):
    return sum([proton.spin for proton in self.protons]) \
        + sum([neutron.spin for neutron in self.neutrons]) \
        + sum([_electron.spin for _electron in self.electrons])

@njit
def momentum(self):
    return np.linalg.norm(self.mass * self.velocity)

@njit
def _debug_mode(self):
    self.debug_mode = True
    for orbital in self._orbitals:
        orbital.debug_mode = True