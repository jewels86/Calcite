import numpy as np
from numba import njit
from calcite.formulas import orbital_order

@njit
def configure(self):
    added = 0
    max_n = int(np.ceil((self.n_electrons / 2)**(1/3)))
    order = orbital_order(self.n_electrons, max_n)
    