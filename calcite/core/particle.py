import numpy as np
from dataclasses import dataclass
from abc import abstractmethod
import calcite

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
    mass: float
    charge: float
    spin: float
    momentum: np.ndarray
    energy: float
    quark_content: dict = {"u": 0, "d": 0, "s": 0, "c": 0, "b": 0, "t": 0}

    def baryon(self):
        

@dataclass
class Electron(Particle):
    def __init__(self, momentum: np.ndarray = np.zeros(3), energy: float = 0):
        super().__init__(
            mass=1.0,
            charge=-1.0,
            spin=0.5,
            momentum=momentum,
            energy=energy,
        )

@dataclass
class Proton(CompositeParticle):
    def __init__(self, momentum: np.ndarray = np.zeros(3), energy: float = 0):
        super().__init__(
            mass=1836.1627,
            charge=1.0,
            spin=0.5,
            momentum=momentum,
            energy=energy,
            quark_content={'u': 2, 'd': 1}
        )

@dataclass
class Neutron(CompositeParticle):
    def __init__(self, momentum: np.ndarray = np.zeros(3), energy: float = 0):
        super().__init__(
            mass=1838.6837,
            charge=0.0,
            spin=0.5,
            momentum=momentum,
            energy=energy,
            quark_content={'u': 1, 'd': 2}
        )
