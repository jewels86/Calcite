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
    
    def register_quark(self, quark: calcite.Quark):
        quark.index = len(self.quarks)
        self.quarks.append(quark)

    def register_particle(self, particle: calcite.Particle):
        particle.index = len(self.particles)
        self.particles.append(particle)
    
    def register_composite(self, composite: calcite.CompositeParticle):
        composite.index = len(self.composites)
        self.composites.append(composite)

    def register_atom(self, atom: calcite.Atom):
        atom.index = len(self.atoms)
        self.atoms.append(atom)
        for proton in atom.protons:
            if proton.index != -1: continue
            self.register_composite(proton)
        for neutron in atom.neutrons:
            if neutron.index != -1: continue
            self.register_composite(neutron)
        for electron in atom.electrons:
            if electron.index != -1: continue
            self.register_particle(electron)

    def register_molecule(self, molecule: calcite.Molecule):
        molecule.index = len(self.molecules)
        self.molecules.append(molecule)
        for atom in molecule.atoms:
            if atom.index != -1: continue
            self.register_atom(atom)

    def register_quarks(self, *quarks):
        for quark in quarks:
            self.register_quark(quark)
    
    def register_particles(self, *particles):
        for particle in particles:
            self.register_particle(particle)

    def register_composites(self, *composites):
        for composite in composites:
            self.register_composite(composite)

    def register_atoms(self, *atoms):
        for atom in atoms:
            self.register_atom(atom)

    def register_molecules(self, *molecules):
        for molecule in molecules:
            self.register_molecule(molecule)

    def debug_mode(self):
        for quark in self.quarks:
            quark.debug_mode = True
        for particle in self.particles:
            particle.debug_mode = True
        for composite in self.composites:
            composite.debug_mode = True
        for atom in self.atoms:
            atom._debug_mode()
        for molecule in self.molecules:
            molecule.debug_mode = True
@njit
def world():
    return World()

world_type = typeof(World())