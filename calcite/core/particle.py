import numpy as np
from dataclasses import dataclass
from abc import abstractmethod
from calcite.core.quark import up_quark, down_quark, Quark

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
    quarks: list[Quark] = []

    def mass(self):
        return sum([quark.mass for quark in self.quarks])
    
    def charge(self):
        return sum([quark.charge for quark in self.quarks])
    
    def spin(self):
        return sum([quark.spin for quark in self.quarks])
    
    def energy(self):
        return self.mass()

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
            quark_content=[
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
            quark_content=[
                up_quark(),
                down_quark(),
                down_quark()
            ]
        )
