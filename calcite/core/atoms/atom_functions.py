import numpy as np
from numba import njit, typed
from calcite.formulas import orbital_order

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