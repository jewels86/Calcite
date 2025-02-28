from numba import float64, int32, types, njit, typeof
from numba.experimental import jitclass

spec = [
    ('type', types.string),
    ('charge', float64),
    ('mass', float64),
    ('spin', float64),
    ('color', types.string)
]

@jitclass(spec)
class Quark:
    def __init__(self, type, charge, mass, spin, color='white'):
        self.type = type
        self.charge = charge
        self.mass = mass
        self.spin = spin
        self.color = color

@njit
def up_quark(color='red'):
    return Quark('up', 2/3, 0.002, 0.5, color)

@njit
def down_quark(color='green'):
    return Quark('down', -1/3, 0.005, 0.5, color)

@njit
def strange_quark(color='blue'):
    return Quark('strange', -1/3, 0.095, 0.5, color)

@njit
def charm_quark(color='cyan'):
    return Quark('charm', 2/3, 1.27, 0.5, color)

@njit
def top_quark(color='magenta'):
    return Quark('top', 2/3, 173.21, 0.5, color)

@njit
def bottom_quark(color='yellow'):
    return Quark('bottom', -1/3, 4.18, 0.5, color)

quark_type = typeof(up_quark())