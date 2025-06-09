#include "Vector3.h"

namespace math {

Vector3::Vector3() : x(0), y(0), z(0) {}

Vector3::Vector3(const double x, const double y, const double z)
    : x(x), y(y), z(z) {}

Vector3::Vector3(const Vector3 &v)
    : x(v.x), y(v.y), z(v.z) {}

Vector3 Vector3::operator+(const Vector3 &v) const {
    return Vector3(x + v.x, y + v.y, z + v.z);
}

Vector3 Vector3::operator-(const Vector3 &v) const {
    return Vector3(x - v.x, y - v.y, z - v.z);
}

Vector3 Vector3::operator*(const double scalar) const {
    return Vector3(x * scalar, y * scalar, z * scalar);
}

Vector3 Vector3::operator/(const double scalar) const {
    if (scalar == 0) {
        throw std::runtime_error("Division by zero in Vector3 division");
    }
    return Vector3(x / scalar, y / scalar, z / scalar);
}

Vector3 Vector3::operator+=(const Vector3 &v) {
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

Vector3 Vector3::operator-=(const Vector3 &v) {
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

Vector3 Vector3::operator*=(const double scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
}

Vector3 Vector3::operator/=(const double scalar) {
    if (scalar == 0) {
        throw std::runtime_error("Division by zero in Vector3 division");
    }
    x /= scalar;
    y /= scalar;
    z /= scalar;
    return *this;
}

Vector3 Vector3::operator-() const {
    return Vector3(-x, -y, -z);
}

bool Vector3::operator==(const Vector3 &v) const {
    return (x == v.x && y == v.y && z == v.z);
}

bool Vector3::operator!=(const Vector3 &v) const {
    return !(*this == v);
}

double Vector3::operator[](const int index) const {
    if (index < 0 || index > 2) {
        throw std::out_of_range("Index out of range in Vector3");
    }
    switch (index) {
        case 0: return x;
        case 1: return y;
        case 2: return z;
        default: throw std::out_of_range("Index out of range in Vector3");
    }
}

double Vector3::operator[](const char index) const {
    switch (index) {
        case 'x': return x;
        case 'y': return y;
        case 'z': return z;
        default: throw std::out_of_range("Index must be 'x', 'y', or 'z' in Vector3");
    }
}

double Vector3::dot(const Vector3 &v) const {
    return x * v.x + y * v.y + z * v.z;
}

Vector3 Vector3::cross(const Vector3 &v) const {
    return Vector3(
        y * v.z - z * v.y,
        z * v.x - x * v.z,
        x * v.y - y * v.x
    );
}

double Vector3::length() const {
    return std::sqrt(x * x + y * y + z * z);
}

Vector3 Vector3::normalize() const {
    const double len = length();
    if (len == 0) {
        throw std::runtime_error("Cannot normalize a zero-length vector");
    }
    return *this / len;
}

} // namespace math