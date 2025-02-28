from dataclasses import dataclass

@dataclass
class Quark:
    type: str
    charge: float
    mass: float
    spin: float
    color: str = 'white'

def up_quark(color='red'):
    return Quark('up', 2/3, 0.002, 0.5, color)

def down_quark(color='green'):
    return Quark('down', -1/3, 0.005, 0.5, color)

def strange_quark(color='blue'):
    return Quark('strange', -1/3, 0.095, 0.5, color)

def charm_quark(color='cyan'):
    return Quark('charm', 2/3, 1.27, 0.5, color)

def top_quark(color='magenta'):
    return Quark('top', 2/3, 173.21, 0.5, color)

def bottom_quark(color='yellow'):
    return Quark('bottom', -1/3, 4.18, 0.5, color)
