from typing import List, Dict, Optional
from calcite.core.molecules.molecule import Molecule
from calcite.core.vectors.vector import Vector

class Aggregate:
    """
    A class representing an aggregate of molecules.

    Attributes:
    - molecules (List[Molecule]): the molecules in the aggregate
    - position (Vector): the position of the aggregate
    - velocity (Vector): the velocity of the aggregate
    - data (Dict[str, float]): additional data associated with the aggregate
    - debug_mode (bool): flag for debug mode

    Methods:
    - add(molecule: Molecule) -> None: adds a molecule to the aggregate
    - mass() -> float: calculates the total mass of the aggregate
    - charge() -> float: calculates the total charge of the aggregate
    - center_of_mass() -> Vector: calculates the center of mass of the aggregate
    """
    molecules: List[Molecule]
    position: Vector
    velocity: Vector
    data: Dict[str, float]
    debug_mode: bool

    def add(self, molecule: Molecule) -> None: ...
    def mass(self) -> float: ...
    def charge(self) -> float: ...
    def center_of_mass(self) -> Vector: ...

def aggregate(
    molecules: Optional[List[Molecule]] = None,
    position: Optional[Vector] = None,
    velocity: Optional[Vector] = None,
    data: Optional[Dict[str, float]] = None,
    debug_mode: bool = False
) -> Aggregate:
    """
    Creates an aggregate.

    Args:
    - molecules (Optional[List[Molecule]]): the molecules in the aggregate
    - position (Optional[Vector]): the position of the aggregate
    - velocity (Optional[Vector]): the velocity of the aggregate
    - data (Optional[Dict[str, float]]): additional data associated with the aggregate
    - debug_mode (bool): flag for debug mode

    Returns:
    - Aggregate: the created aggregate
    """
    ...