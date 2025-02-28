import numpy as np
from dataclasses import dataclass, field
from calcite.core.quark import up_quark, down_quark, Quark
from numba import njit

@dataclass
class Particle:
    mass: float
    "The mass of the particle in atomic mass units."
    charge: float
    spin: float
    momentum: np.ndarray
    energy: float

    color: str = 'white'

@dataclass
class CompositeParticle:
    momentum: np.ndarray
    quarks: list[Quark] = field(default_factory=list)

    @njit
    def mass(self):
        return sum([quark.mass for quark in self.quarks])
    
    @njit
    def charge(self):
        return sum([quark.charge for quark in self.quarks])
    
    @njit
    def spin(self):
        return sum([quark.spin for quark in self.quarks])
    
    @njit
    def energy(self):
        return self.mass()

    @njit
    def baryon(self):
        return len(self.quarks) // 3

@dataclass
class Electron(Particle):
    n: int = 1
    l: int = 0
    m: int = 0

    def __init__(self, momentum: np.ndarray = np.zeros(3), energy: float = 0, 
                 n: int = 1, l: int = 0, m: int = 0):
        super().__init__(
            mass=1.0,
            charge=-1.0,
            spin=0.5,
            momentum=momentum,
            energy=energy,
        )
        self.n = n
        self.l = l
        self.m = m

@dataclass
class Proton(CompositeParticle):
    def __init__(self, momentum: np.ndarray = np.zeros(3)):
        super().__init__(
            momentum=momentum,
            quarks=[
                up_quark(),
                up_quark(),
                down_quark()
            ]
        )

@dataclass
class Neutron(CompositeParticle):
    def __init__(self, momentum: np.ndarray = np.zeros(3)):
        super().__init__(
            momentum=momentum,
            quarks=[
                up_quark(),
                down_quark(),
                down_quark()
            ]
        )

def create_particles(particle_type: type, n: int) -> list:
    return [particle_type() for _ in range(n)]