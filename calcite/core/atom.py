import numpy as np
from dataclasses import dataclass, field
from calcite.core.particle import Electron, Proton, Neutron
import calcite.constants as constants
from numba import njit, float64, int32, types, typed
from numba.experimental import jitclass
import numba


orbital_spec = [
    ('n', int32),
    ('l', int32),
    ('m', int32),
    ('electrons', types.ListType(Electron))
]

@jitclass(orbital_spec)
class Orbital:
    def __init__(self, n, l, m, electrons):
        self.n = n
        self.l = l
        self.m = m
        self.electrons = electrons

    def add(self, spin: float):
        if len(self.electrons) < 2 and spin not in [electron.spin for electron in self.electrons]:
            self.electrons.append(Electron(n=self.n, l=self.l, m=self.m, spin=spin))
            return True
        return False

atom_spec = [
    ('protons', types.ListType(Proton)),
    ('neutrons', types.ListType(Neutron)),
    ('electrons', types.ListType(Electron)),
    ('orbitals', types.DictType(types.Tuple((int32, int32, int32)), int32)),
    ('_orbitals', types.ListType(Orbital))
]

@jitclass(atom_spec)
class Atom:
    def __init__(self, protons, neutrons):
        self.protons = typed.List([Proton() for _ in range(protons)])
        self.neutrons = typed.List([Neutron() for _ in range(neutrons)])
        self.electrons = typed.List.empty_list(Electron)
        self.orbitals = typed.Dict.empty(
            key_type=types.Tuple((int32, int32, int32)),
            value_type=int32
        )
        self._orbitals = typed.List.empty_list(Orbital)

    def configure(self, n_electrons: int):
        order = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0)]
        added = 0
        states = [0.5, -0.5]
        for n, l in order:
            for m in range(-l, l+1):
                if (n, l, m) not in self.orbitals:
                    self.orbitals[(n, l, m)] = len(self._orbitals)
                    self._orbitals.append(Orbital(n=n, l=l, m=m, electrons=[]))
                orbital = self._orbitals[self.orbitals[(n, l, m)]]
                for spin in states:
                    if added < n_electrons and orbital.add(spin):
                        self.electrons.append(orbital.electrons[-1])
                        added += 1
            if added >= n_electrons:
                return
            
    def add_electron(self, electron: Electron):
        for orbital in self.orbitals.values():
            if orbital.add(electron.spin):
                self.electrons.append(electron)
                return True
        return False
    
    def remove_electron(self):
        if self.electrons:
            electron = self.electrons.pop()
            orbital = self.orbitals[(electron.n, electron.l, electron.m)]
            orbital.electrons.remove(electron)
            return electron
        return None
    
    def covalent_bond(self, atom: "Atom"):
        for orbital in self.orbitals.values():
            if len(orbital.electrons) == 1:
                for other in atom.orbitals.values():
                    if len(other.electrons) == 1:
                        orbital.electrons.append(other.electrons[0])
                        other.electrons.append(orbital.electrons[0])
                        return True
        return False

    def ionic_bond(self, atom: "Atom"):
        if self.charge() <= 0 and atom.charge() >= 0:
            electron = self.remove_electron()
            if electron:
                atom.add_electron(electron)
                return True
        return False

    @property
    def mass(self):
        return sum([proton.mass for proton in self.protons]) \
            + sum([neutron.mass for neutron in self.neutrons]) \
            + self.electrons * constants.ELECTRON_MASS
    
    @property
    def atomic_number(self):
        return len(self.protons)
    
    @property
    def charge(self):
        return sum([proton.charge for proton in self.protons]) \
            - self.electrons
    
    @property
    def spin(self):
        return sum([proton.spin for proton in self.protons]) \
            + sum([neutron.spin for neutron in self.neutrons]) \
            + self.electrons * 0.5