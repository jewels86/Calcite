#ifndef MATH_VECTOR3_H
#definte MATH_VECTOR3_H

#pragma once
#include <stdexcept>
#include <cmath>

namespace math {

    class Vector3 {
    public:
        double x, y, z;
        Vector3();
        Vector3(const double x, const double y, const double z);
        Vector3(const Vector3 &v);

        Vector3 operator+(const Vector3 &v) const;
        Vector3 operator-(const Vector3 &v) const;
        Vector3 operator*(const double scalar) const;
        Vector3 operator/(const double scalar) const;
        Vector3 operator+=(const Vector3 &v);
        Vector3 operator-=(const Vector3 &v);
        Vector3 operator*=(const double scalar);
        Vector3 operator/=(const double scalar);
        Vector3 operator-() const;
        bool operator==(const Vector3 &v) const;
        bool operator!=(const Vector3 &v) const;
        double operator[](const int index) const;
        double operator[](const char index) const;

        double dot(const Vector3 &v) const;
        Vector3 cross(const Vector3 &v) const;
        double length() const;
        Vector3 normalize() const;
    };

}

#endif
