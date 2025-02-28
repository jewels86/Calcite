import numpy as np
from dataclasses import dataclass, field
from calcite.core.quark import up_quark, down_quark, Quark
from numba import njit, float64, int32, types
from numba.experimental import jitclass

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
    def __init__(self, mass, charge, spin, momentum, energy, color='white'):
        self.mass = mass
        self.charge = charge
        self.spin = spin
        self.momentum = momentum
        self.energy = energy
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

electron_spec = particle_spec + [
    ('n', int32),
    ('l', int32),
    ('m', int32)
]

@jitclass(electron_spec)
class Electron:
    def __init__(self, momentum=np.zeros(3), energy=0, n=1, l=0, m=0):
        super().__init__(1.0, -1.0, 0.5, momentum, energy)
        self.n = n
        self.l = l
        self.m = m

proton_spec = composite_particle_spec

@jitclass(proton_spec)
class Proton:
    def __init__(self, momentum=np.zeros(3)):
        super().__init__(momentum, [up_quark(), up_quark(), down_quark()])

neutron_spec = composite_particle_spec

@jitclass(neutron_spec)
class Neutron:
    def __init__(self, momentum=np.zeros(3)):
        super().__init__(momentum, [up_quark(), down_quark(), down_quark()])

def create_particles(particle_type: type, n: int) -> list:
    return [particle_type() for _ in range(n)]