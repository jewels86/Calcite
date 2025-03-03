from calcite.core.quark import up_quark, down_quark
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

PROTON_QUARKS = [up_quark(), up_quark(), down_quark()]
NEUTRON_QUARKS = [up_quark(), down_quark(), down_quark()]

LIST = nb.typed.List
LIST_TYPE = nb.types.ListType

NAN_VECTOR = np.array([np.nan, np.nan, np.nan], dtype=np.float64)