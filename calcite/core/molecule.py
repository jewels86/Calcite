from numba.experimental import jitclass
from calcite.core.atom import atom_type
from numba import types, typed, typeof

molecule_spec = [
    ('atoms', types.ListType(atom_type)),
]

@jitclass(molecule_spec)
class Molecule:
    def __init__(self, atoms: list):
        self.atoms = typed.List(atoms)
    
    @property
    def ionic_bonds(self):
        bonds = []
        for atom in self.atoms:
            for bond in atom.ionic_bonds:
                print(2)
                bonds.append(bond)
        return bonds
    
    @property
    def covalent_bonds(self):
        bonds = []
        for atom in self.atoms:
            for bond in atom.covalent_bonds:
                bonds.append(bond)
        return bonds

molecule_type = typeof(Molecule(typed.List.empty_list(atom_type)))