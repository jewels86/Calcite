import numpy as np
from numba import njit, typed, types
from calcite.core.composites.composite import proton, neutron 
from calcite.core.particles.particle import electron, ParticleType
from orbital import Orbital, OrbitalType

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
                    self._orbitals.append(Orbital(n, l, m, typed.List.empty_list(particle_type)))

    to_stay = [orbital for orbital in self._orbitals if len(orbital.electrons) != 0]
    to_remove = [orbital for orbital in self._orbitals if len(orbital.electrons) == 0]
    for orbital in to_remove: self.orbitals.pop((orbital.n, orbital.l, orbital.m), None)
    self._orbitals = typed.List(to_stay)

    if added < len(self.electrons):
        if self.debug_mode:
            print(f"Atom.configure: {len(self.electrons) - added} electrons remain unassigned.")
            print(f"Unassigned electrons will be reconfigured.")
    
    if len(self.electrons) > capacity:
        if self.debug_mode:
            print(f"Atom.configure: Atom has {len(self.electrons)} electrons, but only {capacity} can be assigned.")
            print(f"Excess electrons will be reconfigured.")
    
    if len(unassigned) > 0:
        