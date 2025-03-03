import calcite
from numba import typeof, types, typed
from numba.experimental import jitclass

world_spec = [
    ('quarks', types.ListType(calcite.quark_type)),
    ('particles', types.ListType(calcite.particle_type)),
    ('composites', types.ListType(calcite.composite_particle_type)),
    ('atoms', types.ListType(calcite.atom_type)),
    ('molecules', types.ListType(calcite.molecule_type))
]

@jitclass
class World:
    def __init__(self):
        self.quarks = typed.List.empty_list(calcite.quark_type)
        self.particles = typed.List.empty_list(calcite.particle_type)
        self.composites = typed.List.empty_list(calcite.composite_particle_type)
        self.atoms = typed.List.empty_list(calcite.atom_type)
        self.molecules = typed.List.empty_list(calcite.molecule_type)
    
    def add(self, obj):
        if typeof(obj) == calcite.quark_type:
            self.quarks.append(obj)
        elif typeof(obj) == calcite.particle_type:
            self.particles.append(obj)
        elif typeof(obj) == calcite.composite_particle_type:
            self.composites.append(obj)
        elif typeof(obj) == calcite.atom_type:
            self.atoms.append(obj)
        elif typeof(obj) == calcite.molecule_type:
            self.molecules.append(obj)
        else:
            raise TypeError(f'Invalid object type: {typeof(obj)}')

calcite.world_type = typeof(World())