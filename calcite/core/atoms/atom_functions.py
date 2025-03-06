import numpy as np
from numba import njit, typed, types
from calcite.core.composites.composite import proton, neutron 
from calcite.core.particles.particle import electron, ParticleType
from orbital import Orbital, OrbitalType