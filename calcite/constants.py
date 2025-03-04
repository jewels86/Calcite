import numba as nb
import numpy as np

ELECTRON_FLAG = 0
PROTON_FLAG = 1
NEUTRON_FLAG = 2

ELECTRON_MASS = 1.0
ELECTRON_CHARGE = -1.0
ELECTRON_SPIN = 0.5

PROTON_MASS = 1836.15267389
PROTON_CHARGE = 1.0
PROTON_SPIN = 0.5

NEUTRON_MASS = 1838.68366048
NEUTRON_CHARGE = 0.0
NEUTRON_SPIN = 0.5

LIST = nb.typed.List
LIST_TYPE = nb.types.ListType

NAN_VECTOR = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

ORBITAL_ORDER = nb.typed.List([(n, l) for n in range(1, 7) for l in range(n)])

COLORS = nb.typed.List(['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'white', 'gray', 'purple', 'orange'])