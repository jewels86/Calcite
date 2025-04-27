import numpy as np
from numba.experimental import structref
from numba import njit, types
from numba.extending import overload_method

# region VectorType and Vector
# region Class definitions
@structref.register
class VectorType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)

class Vector(structref.StructRefProxy):
    def __new__(self, x, y, z):
        instance = structref.StructRefProxy.__new__(self, x, y, z)
        return instance

    @property
    def x(self):
        return Vector_get_x(self)

    @x.setter
    def x(self, x):
        Vector_set_x(self, x)

    @property
    def y(self):
        return Vector_get_y(self)

    @y.setter
    def y(self, y):
        Vector_set_y(self, y)

    @property
    def z(self):
        return Vector_get_z(self)

    @z.setter
    def z(self, z):
        Vector_set_z(self, z)
# endregion
# region Vector methods
# region Fields
@njit
def Vector_get_x(self):
    return self.x

@njit
def Vector_get_y(self):
    return self.y

@njit
def Vector_get_z(self):
    return self.z

@njit
def Vector_set_x(self, x):
    self.x = x

@njit
def Vector_set_y(self, y):
    self.y = y

@njit
def Vector_set_z(self, z):
    self.z = z
# endregion
# region Methods

@overload_method(VectorType, 'magnitude')
def Vector_magnitude(self):
    def impl(self):
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    return impl

@overload_method(VectorType, 'normalize')
def Vector_normalize(self):
    def impl(self):
        mag = self.magnitude()
        return Vector(self.x / mag, self.y / mag, self.z / mag)
    return impl

@overload_method(VectorType, 'dot')
def Vector_dot(self, other):
    def impl(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    return impl

@overload_method(VectorType, 'cross')
def Vector_cross(self, other):
    def impl(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    return impl

@overload_method(VectorType, 'angle_with')
def Vector_angle_with(self, other):
    def impl(self, other):
        dot_product = self.dot(other)
        magnitudes = self.magnitude() * other.magnitude()
        return np.arccos(dot_product / magnitudes)
    return impl

@overload_method(VectorType, 'unit')
def Vector_unit(self):
    def impl(self):
        mag = self.magnitude()
        return Vector(self.x / mag, self.y / mag, self.z / mag)
    return impl
# endregion
# endregion

structref.define_proxy(Vector, VectorType, ['x', 'y', 'z'])

vector_type = VectorType(fields=[
    ('x', types.float64),
    ('y', types.float64),
    ('z', types.float64)
])
# endregion
# region Vector creation methods
@njit
def vector(x, y, z) -> Vector:
    """
    Creates a new vector with the given components.

    Args:
        x (float): The x component of the vector.
        y (float): The y component of the vector.
        z (float): The z component of the vector.

    Returns:
        Vector: A new vector object
    """
    return Vector(x, y, z)
# endregion

