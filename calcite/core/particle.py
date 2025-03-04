import numpy as np
from dataclasses import dataclass, field
from calcite.core.quark import up_quark, down_quark, quark_type
from numba import njit, float64, int64, types, typed, typeof
from numba.experimental import jitclass
from calcite import  constants

quark_type = typeof(up_quark())

particle_spec = [
    ('mass', float64),
    ('charge', float64),
    ('spin', float64),
    ('position', float64[:]),
    ('velocity', float64[:]),
    ('energy', float64),
    ('data', types.DictType(types.unicode_type, float64)),
    ('index', int64),
    ('debug_mode', types.boolean)
]

@jitclass(particle_spec)
class Particle:
    """
    A class representing a particle in the Standard Model of particle physics.

    Attributes:
    - mass (float): the mass of the particle in atomic units
    - charge (float): the electric charge of the particle
    - spin (float): the spin of the particle
    - position (np.ndarray): the position of the particle in 3D space (may be NaN)
    - velocity (np.ndarray): the velocity of the particle in 3D space (may be NaN)
    - energy (float): the energy of the particle in Hartrees
    - data (dict): additional data about the particle

    Properties:
    - momentum (np.ndarray): the momentum of the particle
    - kinetic_energy (float): the kinetic energy of the particle
    """
    def __init__(self, mass: float, charge: float, spin: float, position: list[float] | None, 
                 velocity: list[float] | None, energy: float, data: dict[str, object]):
        """
        Initialize a Particle object with the given mass, charge, spin, position, velocity, energy, and data.

        Args:
        - mass (float): the mass of the particle in atomic units
        - charge (float): the electric charge of the particle
        - spin (float): the spin of the particle
        - position (list[float]): the position of the particle in 3D space
        - velocity (list[float]): the velocity of the particle in 3D space
        - energy (float): the energy of the particle in Hartrees
        - data (dict): additional data about the particle

        Returns:
        - Particle: a Particle object with the specified attributes
        """
        nan_array = np.full(3, np.nan, dtype=np.float64)

        self.mass: float = mass
        self.charge = charge
        self.spin = spin
        self.position = np.array(position, dtype=np.float64) if position is not None else nan_array
        self.velocity = np.array(velocity, dtype=np.float64) if velocity is not None else nan_array
        self.energy = energy if energy > 0 else self.mass
        self.data = data
        self.debug_mode = False

    @property
    def momentum(self):
        return self.mass * self.velocity
    
    @property
    def kinetic_energy(self):
        return 0.5 * self.mass * np.linalg.norm(self.velocity) ** 2

composite_particle_spec = [
    ('position', float64[:]),
    ('velocity', float64[:]),
    ('quarks', types.ListType(quark_type)),
    ('data', types.DictType(types.unicode_type, float64)),
    ('index', int64),
    ('debug_mode', types.boolean)
]

@jitclass(composite_particle_spec)
class CompositeParticle:
    def __init__(self, position, velocity, quarks, data):
        nan_array = np.full(3, np.nan, dtype=np.float64)

        self.position = position if position is not None else nan_array
        self.velocity = velocity if velocity is not None else nan_array
        self.quarks = quarks
        self.data = data
        self.debug_mode = False

    @property
    def mass(self):
        return sum([quark.mass for quark in self.quarks]) + 0.0103
    
    @property
    def charge(self):
        return sum([quark.charge for quark in self.quarks])
    
    @property
    def spin(self):
        return sum([quark.spin for quark in self.quarks])
    
    @property
    def energy(self):
        return self.mass

    @property
    def baryon(self):
        return len(self.quarks) // 3
    
    @property
    def momentum(self):
        return self.mass * self.velocity

@njit
def electron(n: int, l: int, m: int, position: list[float] | None = None, 
             velocity: list[float] = None, energy: float = -1) -> Particle:
    data = typed.Dict.empty(types.unicode_type, float64)
    data['type'] = constants.ELECTRON_FLAG
    data['n'] = n
    data['l'] = l
    data['m'] = m

    return Particle(
        constants.ELECTRON_MASS, 
        constants.ELECTRON_CHARGE, 
        constants.ELECTRON_SPIN, 
        position, 
        velocity, 
        energy, 
        data
    )

@njit
def proton(position: list[float] | None = None, velocity: list[float] | None = None) -> CompositeParticle:
    data = typed.Dict.empty(types.string, float64)
    data['type'] = constants.PROTON_FLAG
    return CompositeParticle(
        position,
        velocity,
        typed.List([up_quark(), up_quark(), down_quark()]),
        data
    )

@njit
def neutron(position: list[float] | None = None, velocity: list[float] | None = None) -> CompositeParticle:
    data = typed.Dict.empty(types.string, float64)
    data['type'] = constants.NEUTRON_FLAG
    return CompositeParticle(
        position,
        velocity,
        typed.List([up_quark(), down_quark(), down_quark()]),
        data
    )

particle_type = typeof(electron(1, 0, 1))
composite_particle_type = typeof(proton())