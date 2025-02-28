import numpy as np
from dataclasses import dataclass
from enum import Enum
from calcite.core.formulas import energy

class ParticleType(Enum):
    """Enum for particle types."""
    ELECTRON = 1
    PROTON = 2
    NEUTRON = 3

@dataclass
class Particle:
    mass: float
    "The mass of the particle in GeV/c^2."
    
    charge: float
    "The charge of the particle in Coulombs."

    momentum: np.ndarray
    "The momentum of the particle in GeV/c."

    energy: float
    "The energy of the particle in GeV."

    velocity: np.ndarray
    "The velocity of the particle in m/s."

    position: np.ndarray
    "The position of the particle in fm."

    positions: np.ndarray
    "The positions of the particle in fm."
    velocities: np.ndarray
    "The velocities of the particle in m/s."
    momenta: np.ndarray
    "The momenta of the particle in GeV/c."
    energies: np.ndarray
    "The energies of the particle in GeV."

    type: ParticleType
    "The type of the particle."

    def __init__(self, mass: float, charge: float, type: ParticleType, dt: float, tf: float, momentum: np.ndarray = None, 
                 energy: float = None, velocity: np.ndarray = None, position: np.ndarray = None,):
        self.mass = mass
        self.charge = charge
        self.momentum = momentum if momentum is not None else np.zeros(3)
        self.energy = energy if energy is not None else np.sqrt(np.sum(self.momentum**2) + self.mass**2)
        self.velocity = velocity if velocity is not None else np.zeros(3)
        self.position = position if position is not None else np.zeros(3)
        self.positions = np.zeros((tf/dt, 3))
        self.velocities = np.zeros((tf/dt, 3))
        self.momenta = np.zeros((tf/dt, 3))
        self.energies = np.zeros(tf/dt)
        self.type = type
        self.dt = dt
        self.tf = tf

    def update_position(self, position: np.ndarray, time: float) -> None:
        """Update the position of the particle."""
        self.position = position
        self.positions[time/self.dt]
    
    def update_velocity(self, velocity: np.ndarray, time: float) -> None:
        """Update the velocity of the particle."""
        self.velocity = velocity
        self.velocities[time/self.dt]
        self.update_momentum(velocity * self.mass, time)
        self.update_energy(energy(self.mass, self.momentum), time)

    def update_momentum(self, momentum: np.ndarray, time: float) -> None:
        """Update the momentum of the particle."""
        self.momentum = momentum
        self.momenta[time/self.dt]
    
    def update_energy(self, energy: float, time: float) -> None:
        """Update the energy of the particle."""
        self.energy = energy
        self.energies[time/self.dt]