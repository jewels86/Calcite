from typing import List, Dict, Tuple, Optional
from calcite.core.particles.particle import Particle
from calcite.core.vectors.vector import Vector
from calcite.core.atoms.orbital import Orbital

class Atom:
    """
    A class representing an atom.

    Attributes:
    - protons (List[Particle]): the protons in the atom
    - neutrons (List[Particle]): the neutrons in the atom
    - electrons (List[Particle]): the electrons in the atom
    - ref_orbitals (Dict[Tuple[int, int, int], int]): reference to orbitals by quantum numbers
    - orbitals (List[Orbital]): the orbitals in the atom
    - ionic_bonds (List[Tuple[int, int]]): ionic bonds of the atom
    - covalent_bonds (List[Tuple[int, int, int]]): covalent bonds of the atom
    - position (Vector): the position of the atom
    - velocity (Vector): the velocity of the atom
    - data (Dict[str, str]): additional data associated with the atom
    - index (int): the index of the atom
    - debug_mode (bool): flag for debug mode
    - n_electrons (int): the number of electrons in the atom

    Methods:
    - configure(): configures the atom's orbitals and electrons
    - add(electron: Particle) -> bool: adds an electron to the atom
    - remove() -> bool: removes an electron from the atom
    - remove_specific(electron: Particle) -> bool: removes a specific electron from the atom
    - valence_electrons() -> List[Particle]: returns the valence electrons of the atom
    - stable() -> bool: checks if the atom is stable
    - add_to_valence_shell(electron: Particle) -> bool: adds an electron to the valence shell
    - remove_from_valence_shell() -> bool: removes an electron from the valence shell
    - covalent_bond(other: "Atom") -> bool: forms a covalent bond with another atom
    - ionic_bond(other: "Atom") -> bool: forms an ionic bond with another atom
    - kinetic_energy() -> float: calculates the kinetic energy of the atom
    - momentum() -> Vector: calculates the momentum of the atom
    """
    protons: List[Particle]
    neutrons: List[Particle]
    electrons: List[Particle]
    ref_orbitals: Dict[Tuple[int, int, int], int]
    orbitals: List[Orbital]
    ionic_bonds: List[Tuple[int, int]]
    covalent_bonds: List[Tuple[int, int, int]]
    position: Vector
    velocity: Vector
    data: Dict[str, str]
    index: int
    debug_mode: bool
    n_electrons: int

    def configure(self) -> None: ...
    def add(self, electron: Particle) -> bool: ...
    def remove(self) -> bool: ...
    def remove_specific(self, electron: Particle) -> bool: ...
    def valence_electrons(self) -> List[Particle]: ...
    def stable(self) -> bool: ...
    def add_to_valence_shell(self, electron: Particle) -> bool: ...
    def remove_from_valence_shell(self) -> bool: ...
    def covalent_bond(self, other: "Atom") -> bool: ...
    def ionic_bond(self, other: "Atom") -> bool: ...
    def kinetic_energy(self) -> float: ...
    def momentum(self) -> Vector: ...

def atom(n_protons: int, n_neutrons: int, n_electrons: int, 
         position: Optional[Vector] = None, 
         velocity: Optional[Vector] = None, 
         debug_mode: bool = False, 
         log: Optional[str] = None) -> Atom:
    """
    Creates an atom.

    Args:
    - n_protons (int): the number of protons
    - n_neutrons (int): the number of neutrons
    - n_electrons (int): the number of electrons
    - position (Optional[Vector]): the position of the atom
    - velocity (Optional[Vector]): the velocity of the atom
    - debug_mode (bool): flag for debug mode
    - log (Optional[str]): log information

    Returns:
    - Atom: the created atom
    """
    ...

class AtomType:
    pass
class atom_type:
    pass