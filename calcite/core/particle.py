import numpy as np
from dataclasses import dataclass, field
from calcite.core.quark import up_quark, down_quark, Quark
from calcite.constants import PROTON_QUARKS, NEUTRON_QUARKS
from numba import njit, float64, int32, types, typed, typeof
from numba.experimental import jitclass

quark_type = typeof(up_quark())

particle_spec = [
    ('mass', float64),
    ('charge', float64),
    ('spin', float64),
    ('momentum', float64[:]),
    ('energy', float64),
    ('color', types.string)
]

@jitclass(particle_spec)
class Particle:
    def __init__(self, mass: float, charge: float, spin: float, momentum: np.ndarray = None, energy: float = 0, color='white'):
        self.mass = mass
        self.charge = charge
        self.spin = spin
        self.momentum = momentum if momentum is not None else np.zeros(3)
        self.energy = energy if energy > 0 else self.mass
        self.color = color

composite_particle_spec = [
    ('momentum', float64[:]),
    ('quarks', types.ListType(Quark))
]

@jitclass(composite_particle_spec)
class CompositeParticle:
    def __init__(self, momentum, quarks):
        self.momentum = momentum
        self.quarks = quarks

    @property
    def mass(self):
        return sum([quark.mass for quark in self.quarks])
    
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

electron_spec = particle_spec + [
    ('n', int32),
    ('l', int32),
    ('m', int32)
]

@jitclass(electron_spec)
class Electron:
    def __init__(self, momentum=np.zeros(3), energy=0, n=1, l=0, m=0):
        self.mass = 1.0
        self.charge = -1.0
        self.spin = 0.5
        self.momentum = momentum
        self.energy = energy if energy > 0 else self.mass
        self.color = 'white'

        self.n = n
        self.l = l
        self.m = m

proton_spec = composite_particle_spec

@jitclass(proton_spec)
class Proton:
    def __init__(self, momentum=np.zeros(3)):
        self.momentum = momentum
        self.quarks = typed.List.empty_list(quark_type)
        self.quarks.append(up_quark())
        self.quarks.append(up_quark())
        self.quarks.append(down_quark())

neutron_spec = composite_particle_spec

@jitclass(neutron_spec)
class Neutron:
    def __init__(self, momentum=np.zeros(3)):
        self.momentum = momentum
        self.quarks = typed.List.empty_list(quark_type)
        self.quarks.append(up_quark())
        self.quarks.append(down_quark())
        self.quarks.append(down_quark())