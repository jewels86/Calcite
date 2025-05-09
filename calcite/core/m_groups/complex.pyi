from typing import List, Dict, Optional
from calcite.core.molecules.molecule import Molecule
from calcite.core.vectors.vector import Vector

class Complex:
    """
    A class representing a complex of molecules.

    Attributes:
    - molecules (List[Molecule]): the molecules in the complex
    - position (Vector): the position of the complex
    - velocity (Vector): the velocity of the complex
    - data (Dict[str, float]): additional data associated with the complex
    - debug_mode (bool): flag for debug mode

    Methods:
    - add(molecule: Molecule) -> None: adds a molecule to the complex
    - mass() -> float: calculates the total mass of the complex
    - charge() -> float: calculates the total charge of the complex
    - center_of_mass() -> Vector: calculates the center of mass of the complex
    - intermolecular_bonds() -> List[tuple]: retrieves intermolecular bonds in the complex
    - structure() -> Dict[str, int]: retrieves the structure of the complex
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
    def intermolecular_bonds(self) -> List[tuple]: ...
    def structure(self) -> Dict[str, int]: ...

def complex_(
    molecules: Optional[List[Molecule]] = None,
    position: Optional[Vector] = None,
    velocity: Optional[Vector] = None,
    data: Optional[Dict[str, float]] = None,
    debug_mode: bool = False
) -> Complex:
    """
    Creates a complex.

    Args:
    - molecules (Optional[List[Molecule]]): the molecules in the complex
    - position (Optional[Vector]): the position of the complex
    - velocity (Optional[Vector]): the velocity of the complex
    - data (Optional[Dict[str, float]]): additional data associated with the complex
    - debug_mode (bool): flag for debug mode

    Returns:
    - Complex: the created complex
    """
    ...