import numpy as np

class VectorType:
    pass

class Vector:
    """
    A class representing a 3D vector.

    Attributes:
    - x (float): The x component of the vector.
    - y (float): The y component of the vector.
    - z (float): The z component of the vector.

    Methods:
    - magnitude(): returns the magnitude of the vector.
    - normalize(): returns the normalized vector.
    - dot(other: Vector): returns the dot product with another vector.
    - cross(other: Vector): returns the cross product with another vector.
    - angle_with(other: Vector): returns the angle with another vector.
    - unit_x(): returns the unit vector along the x-axis.
    - unit_y(): returns the unit vector along the y-axis.
    - unit_z(): returns the unit vector along the z-axis.
    """
    x: float
    """The x component of the vector."""
    y: float
    """The y component of the vector."""
    z: float
    """The z component of the vector."""

    def magnitude(self) -> float: ...
    """Returns the magnitude of the vector."""
    def normalize(self) -> 'Vector': ...
    """Returns the normalized vector."""
    def dot(self, other: 'Vector') -> float: ...
    """Returns the dot product with another vector."""
    def cross(self, other: 'Vector') -> 'Vector': ...
    """Returns the cross product with another vector."""
    def angle_with(self, other: 'Vector') -> float: ...
    """Returns the angle with another vector."""
    @staticmethod
    def unit_x() -> 'Vector': ...
    """Returns the unit vector along the x-axis."""
    @staticmethod
    def unit_y() -> 'Vector': ...
    """Returns the unit vector along the y-axis."""
    @staticmethod
    def unit_z() -> 'Vector': ...
    """Returns the unit vector along the z-axis."""

def vector(x: float, y: float, z: float) -> Vector: ...
"""
Creates a new vector with the given components.

Args:
    x (float): The x component of the vector.
    y (float): The y component of the vector.
    z (float): The z component of the vector.

Returns:
    Vector: A new vector object
"""

vector_type = VectorType
"""The type of a vector object."""
