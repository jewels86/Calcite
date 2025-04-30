import numpy as np
from numba import njit, typed, types
from calcite.formulas import orbital_order
from calcite.core.particles.particle import particle_type, electron
from calcite.core.atoms.orbital import orbital, orbital_type


orbital_key_type = types.Tuple((types.int64, types.int64, types.int64))

@njit(cache=True)
def configure(self):
    added = 0
    order = orbital_order(self.n_electrons)

    self.orbitals = typed.List.empty_list(orbital_type)
    self.ref_orbitals = typed.Dict.empty(orbital_key_type, types.int64)
    
    for n, l in order:
        for m in range(-l, l + 1):
            o = orbital(n, l, m, typed.List.empty_list(particle_type))
            self.orbitals.append(o)
            self.ref_orbitals[(n, l, m)] = len(self.orbitals) - 1
            for _ in range(2):
                    if added < self.n_electrons:
                        e = electron(n, l, m, o.open_spin())
                        if o.add(e):
                            self.electrons.append(e)
                            if self.debug_mode: 
                                print(f"Assigned electron with spin {'up' if e.spin == 0.5 else 'down'} to orbital {n}, {l}, {m}")
                            added += 1
                    else:
                        break

@njit
def add(self, electron):
    for o in self.orbitals:
        if o.can_add(electron):
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
    max_n = max([o.n for o in self.orbitals])

    for o in self.orbitals:
        if o.n == max_n:
            valence_electrons.extend(o.electrons)

    return valence_electrons

@njit
def stable(self):
    valence_electrons = 0
    max_valence = 8 if self.n_electrons < 18 else 18
    max_n = max([o.n for o in self.orbitals])

    for o in self.orbitals:
        if o.n == max_n:
            valence_electrons += len(o.electrons)
    
    return valence_electrons == max_valence or valence_electrons == 0 or valence_electrons == 2

@njit
def add_to_valence_shell(self, electron):
    valence_shell = max([o.n for o in self.orbitals])
    for o in self.orbitals:
        if o.n == valence_shell and o.open_spin() != -1.0:
            electron.spin = o.open_spin()
            electron.data["n"] = float(valence_shell)
            electron.data["l"] = float(o.l)
            electron.data["m"] = float(o.m)
            if o.add(electron):
                self.electrons.append(electron)
                return True
            else:
                print("Atom.add_to_valence_shell: Failed to add electron to existing orbital.")
                return False
    
    for l in range(valence_shell):
        for m in range(-l, l + 1):
            if (valence_shell, l, m) not in self.ref_orbitals:
                new_orbital = orbital(valence_shell, l, m, typed.List.empty_list(particle_type))
                self.orbitals.append(new_orbital)
                self.ref_orbitals[(valence_shell, l, m)] = len(self.orbitals) - 1
                if new_orbital.can_add(electron):
                    new_orbital.add(electron)
                    self.electrons.append(electron)
                    return True
    return False

@njit
def remove_from_valence_shell(self):
    valence_shell = max([o.n for o in self.orbitals])
    for o in self.orbitals:
        if o.n == valence_shell and len(o.electrons) > 0:
            o.remove()
            self.electrons.pop()
            return True
    return False

@njit
def covalent_bond(self, other):
    if self.stable() or other.stable():
        print("Atom.covalent_bond: One of the atoms is stable.")
        return False

    valence_self = self.valence_electrons()
    valence_other = other.valence_electrons()

    unpaired_self = [e for e in valence_self]
    unpaired_other = [e for e in valence_other]

    if len(unpaired_self) == 0 or len(unpaired_other) == 0:
        print("Atom.covalent_bond: No unpaired electrons.")
        return False

    electron_self = unpaired_self[0]
    electron_other = unpaired_other[0]

    if not self.add_to_valence_shell(electron_other) or not other.add_to_valence_shell(electron_self):
        print("Atom.covalent_bond: Failed to add electrons to valence shell.")
        return False

    self.covalent_bonds.append((other.index, electron_self.index, electron_other.index))
    other.covalent_bonds.append((self.index, electron_other.index, electron_self.index))

    return True

@njit
def ionic_bond(self, other):
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