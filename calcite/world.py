import calcite
from numba import typeof, types, typed, njit
from numba.experimental import jitclass

world_spec = [
    ('quarks', types.ListType(calcite.quark_type)),
    ('particles', types.ListType(calcite.particle_type)),
    ('composites', types.ListType(calcite.composite_particle_type)),
    ('atoms', types.ListType(calcite.atom_type)),
    ('molecules', types.ListType(calcite.molecule_type))
]

@jitclass(world_spec)
class World:
    def __init__(self):
        self.quarks = typed.List.empty_list(calcite.quark_type)
        self.particles = typed.List.empty_list(calcite.particle_type)
        self.composites = typed.List.empty_list(calcite.composite_particle_type)
        self.atoms = typed.List.empty_list(calcite.atom_type)
        self.molecules = typed.List.empty_list(calcite.molecule_type)
    
    def register(self, obj):
        if typeof(obj) == calcite.quark_type:
            obj.index = len(self.quarks)
            self.quarks.append(obj)

        elif typeof(obj) == calcite.particle_type:
            obj.index = len(self.particles)
            self.particles.append(obj)

        elif typeof(obj) == calcite.composite_particle_type:
            obj.index = len(self.composites)
            self.composites.append(obj)

        elif typeof(obj) == calcite.atom_type:
            obj.index = len(self.atoms)
            self.atoms.append(obj)

        elif typeof(obj) == calcite.molecule_type:
            obj.index = len(self.molecules)
            self.molecules.append(obj)
        else:
            raise TypeError(f'Invalid object type: {typeof(obj)}')

    def registers(self, *objs):
        for obj in objs:
            self.register(obj)
@njit
def world():
    return World()

world_type = typeof(World())