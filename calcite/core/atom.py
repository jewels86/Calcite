import numpy as np
from dataclasses import dataclass, field
from calcite.core.particle import Electron, Proton, Neutron
import calcite.constants as constants
from numba import njit, typed

@dataclass
class Orbital:
    n: int
    l: int
    m: int
    electrons: typed.List[Electron]
    first: bool = False

    @njit
    def add(self, spin: float):
        if len(self.electrons) < 2 and spin not in [electron.spin for electron in self.electrons]:
            self.electrons.append(Electron(n=self.n, l=self.l, m=self.m, spin=spin))
            return True
        return False

@dataclass
class Atom:
    protons: typed.List[Proton]
    neutrons: typed.List[Neutron]
    electrons: typed.List[Electron] = field(default_factory=typed.List)
    orbitals: typed.Dict[tuple[int, int, int], Orbital] = field(default_factory=typed.Dict)

    @njit
    def configure(self, n_electrons: int):
        order = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0)]
        added = 0
        states = [0.5, -0.5]
        for n, l in order:
            for m in range(-l, l+1):
                if (n, l, m) not in self.orbitals:
                    self.orbitals[(n, l, m)] = Orbital(n=n, l=l, m=m, electrons=[])
                orbital = self.orbitals[(n, l, m)]
                for spin in states:
                    if added < n_electrons and orbital.add(spin):
                        self.electrons.append(orbital.electrons[-1])
                        added += 1
            if added >= n_electrons:
                return
            
    @njit
    def add_electron(self, electron: Electron):
        for orbital in self.orbitals.values():
            if orbital.add(electron.spin):
                self.electrons.append(electron)
                return True
        return False
    
    @njit
    def remove_electron(self):
        if self.electrons:
            electron = self.electrons.pop()
            orbital = self.orbitals[(electron.n, electron.l, electron.m)]
            orbital.electrons.remove(electron)
            return electron
        return None
    
    @njit
    def covalent_bond(self, atom: "Atom"):
        for orbital in self.orbitals.values():
            if len(orbital.electrons) == 1:
                for other in atom.orbitals.values():
                    if len(other.electrons) == 1:
                        orbital.electrons.append(other.electrons[0])
                        other.electrons.append(orbital.electrons[0])
                        return True
        return False

    @njit
    def ionic_bond(self, atom: "Atom"):
        if self.charge() <= 0 and atom.charge() >= 0:
            electron = self.remove_electron()
            if electron:
                atom.add_electron(electron)
                return True
        return False

    @njit
    def mass(self):
        return sum([proton.mass for proton in self.protons]) \
            + sum([neutron.mass for neutron in self.neutrons]) \
            + self.electrons * constants.ELECTRON_MASS
    
    @njit
    def atomic_number(self):
        return len(self.protons)
    
    @njit
    def charge(self):
        return sum([proton.charge for proton in self.protons]) \
            - self.electrons
    
    @njit
    def spin(self):
        return sum([proton.spin for proton in self.protons]) \
            + sum([neutron.spin for neutron in self.neutrons]) \
            + self.electrons * 0.5