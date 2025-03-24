import numpy as np
from numba import njit, typed
from calcite.formulas import orbital_order
from calcite.core.particles.particle import particle_type
from calcite.core.atoms.orbital import orbital

@njit
def configure(self):
    added = 0
    max_n = int(np.ceil((self.n_electrons / 2)**(1/3)))
    order = orbital_order(self.n_electrons, max_n)
    if self.debug_mode:
        self._debug("Atom: configure()", 0, f"Configuring atom with {self.n_electrons} electrons and max_n {max_n}.")

    for orbital in self.orbitals:
        orbital.electrons.clear()
    
    for n, l in order:
        for m in range(-l, l+1):
            if (n, l, m) not in self.ref_orbitals:
                self.ref_orbitals[(n, l, m)] = len(self.orbitals)
                self.orbitals.append(orbital(n, l, m))
            o = self.orbitals[self.ref_orbitals[(n, l, m)]]
            for _ in range(2):
                if added < self.n_electrons:
                    e = self.electrons[added]
                    if o.add(e):
                        added += 1
                    else: break
                else: break
    
    to_stay = [orbital for orbital in self._orbitals if len(orbital.electrons) != 0]
    to_remove = [orbital for orbital in self._orbitals if len(orbital.electrons) == 0]
    for orbital in to_remove: self.orbitals.pop((orbital.n, orbital.l, orbital.m), None)
    self._orbitals = typed.List(to_stay)

@njit
def add(self, electron):
    for orbital in self.orbitals:
        if orbital.can_add(electron):
            self.electrons.append(electron)
            self.configure()
            return True
    return False

@njit
def remove(self):
    if len(self.electrons) > 0:
        self.electrons.pop()
        self.configure()
        return True
    return False

@njit
def remove_specific(self, electron):
    if electron.index == -1:
        return False
    new_electrons = typed.List.empty_list(particle_type)
    removed = False

    for e in self.electrons:
        if e.index != electron.index or removed:
            new_electrons.append(e)
        else:
            removed = True

    if removed:
        self.electrons = new_electrons
        self.configure()
        return True
    return False

@njit
def valence_electrons(self):
    valence_electrons = []
    max_n = max([o.n for o in self._orbitals])

    for orbital in self._orbitals:
        if orbital.n == max_n:
            valence_electrons.extend(orbital.electrons)

    return valence_electrons

@njit
def stable(self):
    valence_electrons = 0
    max_valence = 8 if self.atomic_number < 18 else 18
    max_n = max([o.n for o in self._orbitals])

    for o in self.orbitals:
        if o.n == max_n:
            valence_electrons += len(o.electrons)
    
    return valence_electrons == max_valence or valence_electrons == 0 or valence_electrons == 2

@njit
def add_to_valence_shell(self, electron):
    valence_shell = max([o.n for o in self._orbitals])
    for orbital in self._orbitals:
        if orbital.n == valence_shell and orbital.can_add(electron):
            orbital.add(electron)
            self.electrons.append(electron)
            return True
    
    for l in range(valence_shell):
        for m in range(-l, l + 1):
            if (valence_shell, l, m) not in self.orbitals:
                self.orbitals[(valence_shell, l, m)] = len(self._orbitals)
                new_orbital = orbital(valence_shell, l, m, typed.List.empty_list(particle_type))
                self._orbitals.append(new_orbital)
                if new_orbital.can_add(electron):
                    new_orbital.add(electron)
                    self.electrons.append(electron)
                    return True
    
    return False

@njit
def remove_from_valence_shell(self):
    valence_shell = max([o.n for o in self._orbitals])
    for orbital in self._orbitals:
        if orbital.n == valence_shell and len(orbital.electrons) > 0:
            orbital.remove()
            self.electrons.pop()
            return True
    return False

@njit
def covalent_bond(self, other):
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

    if not self.add_to_valence_shell(unpaired_other[0]) or \
        not other.add_to_valence_shell(unpaired_self[0]):
        unpaired_self[0].spin = 0.5
        unpaired_other[0].spin = -0.5
        if not self.add_to_valence_shell(unpaired_other[0]) or \
            not other.add_to_valence_shell(unpaired_self[0]):
            unpaired_self[0].spin = original_self_spin
            unpaired_other[0].spin = original_other_spin
            return False
    
    self.covalent_bonds.append((other.index, unpaired_self[0].index, unpaired_other[0].index))
    other.covalent_bonds.append((self.index, unpaired_other[0].index, unpaired_self[0].index))

    return True
