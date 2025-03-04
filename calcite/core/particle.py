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
        Initializes a Particle object with the given mass, charge, spin, position, velocity, energy, and data.

        Args:
        - mass (float): the mass of the particle in atomic units
        - charge (float): the electric charge of the particle
        - spin (float): the spin of the particle
        - position (list[float]): the position of the particle in 3D space
        - velocity (list[float]): the velocity of the particle in 3D space
        - energy (float): the energy of the particle in Hartrees
        - data (dict): additional data about the particle

        Returns:
            Particle: a Particle object with the specified attributes
        """
        nan_array = np.full(3, np.nan, dtype=np.float64)

        self.mass: float = mass
        "The mass of the particle in atomic units."
        self.charge: float = charge
        "The electric charge of the particle."
        self.spin: float = spin
        "The spin of the particle."
        self.position: np.ndarray = np.array(position, dtype=np.float64) if position is not None else nan_array
        "The position of the particle in 3D space (may be NaN)."
        self.velocity: np.ndarray = np.array(velocity, dtype=np.float64) if velocity is not None else nan_array
        "The velocity of the particle in 3D space (may be NaN)."
        self.energy: float = energy if energy > 0 else self.mass
        "The energy of the particle in Hartrees."
        self.data: dict = data
        "Additional data about the particle."
        self.debug_mode: bool = False

    @property
    def momentum(self):
        "The momentum of the particle (m * v)."
        return self.mass * self.velocity
    
    @property
    def kinetic_energy(self):
        "The kinetic energy of the particle (0.5 * m * v^2)."
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
    """
    A class representing a composite particle made up of quarks.

    Attributes:
    - position (np.ndarray): the position of the composite particle in 3D space (may be NaN)
    - velocity (np.ndarray): the velocity of the composite particle in 3D space (may be NaN)
    - quarks (list): a list of quarks that make up the composite particle
    - data (dict): additional data about the composite particle

    Properties:
    - mass (float): the mass of the composite particle
    - charge (float): the electric charge of the composite particle
    - spin (float): the spin of the composite particle
    - energy (float): the energy of the composite particle
    - baryon (int): the baryon number of the composite particle
    - momentum (np.ndarray): the momentum of the composite particle
    """
    def __init__(self, position, velocity, quarks, data):
        """
        Initializes a CompositeParticle object with the given position, velocity, quarks, and data.

        Args:
        - position (list[float]): the position of the composite particle in 3D space
        - velocity (list[float]): the velocity of the composite particle in 3D space
        - quarks (list): a list of quarks that make up the composite particle
        - data (dict): additional data about the composite particle

        Returns:
            CompositeParticle: a CompositeParticle object with the specified attributes
        """
        nan_array = np.full(3, np.nan, dtype=np.float64)

        self.position = position if position is not None else nan_array
        "The position of the composite particle in 3D space (may be NaN)."
        self.velocity = velocity if velocity is not None else nan_array
        "The velocity of the composite particle in 3D space (may be NaN)."
        self.quarks = quarks
        "A list of quarks that make up the composite particle."
        self.data = data
        "Additional data about the composite particle."
        self.debug_mode = False

    @property
    def mass(self):
        "The mass of the composite particle."
        return sum([quark.mass for quark in self.quarks]) + 0.0103
    
    @property
    def charge(self):
        "The electric charge of the composite particle."
        return sum([quark.charge for quark in self.quarks])
    
    @property
    def spin(self):
        "The spin of the composite particle."
        return sum([quark.spin for quark in self.quarks])
    
    @property
    def energy(self):
        "The energy of the composite particle."
        return self.mass

    @property
    def baryon(self):
        "The baryon number of the composite particle."
        return len(self.quarks) // 3
    
    @property
    def momentum(self):
        "The momentum of the composite particle (m * v)."
        return self.mass * self.velocity

@njit
def electron(n: int, l: int, m: int, position: list[float] | None = None, 
             velocity: list[float] = None, energy: float = -1) -> Particle:
    """
    Creates a new electron with the given quantum numbers.

    Args:
        n (int): The principal quantum number.
        l (int): The azimuthal quantum number.
        m (int): The magnetic quantum number.
        position (list[float] | None, optional): The position of the electron. Defaults to None.
        velocity (list[float], optional): The velocity of the electron. Defaults to None.
        energy (float, optional): The energy of the electron. Defaults to -1.

    Returns:
        Particle: A new electron object
    """
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
    """
    Creates a new proton with the given position and velocity.

    Args:
        position (list[float] | None, optional): The position of the proton. Defaults to None.
        velocity (list[float] | None, optional): The velocity of the proton. Defaults to None.

    Returns:
        CompositeParticle: A new proton object
    """
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
    """
    Creates a new neutron with the given position and velocity.

    Args:
        position (list[float] | None, optional): The position of the neutron. Defaults to None.
        velocity (list[float] | None, optional): The velocity of the neutron. Defaults to None.

    Returns:
        CompositeParticle: A new neutron object
    """
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