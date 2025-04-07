import numpy as np
from numba import njit

@njit
def orbital_order(n_electrons: int, max_n=-1) -> list[tuple[int, int]]:
    max_n = int(np.ceil((n_electrons / 2)**(1/3))) if max_n == -1 else max_n
    return [(n, l) for n in range(1, max_n+1) for l in range(n)]