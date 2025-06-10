#ifndef MATH_VECTOR3_H
#define MATH_VECTOR3_H

#pragma once
#include <stdexcept>
#include <cmath>

#include "ScaledDouble.h"

namespace math {

    class Vector3 {
    public:
        double_sc x, y, z;
        Vector3();
        Vector3(const double_sc x, const double_sc y, const double_sc z);
        Vector3(const Vector3 &v);

        Vector3 operator+(const Vector3 &v) const;
        Vector3 operator-(const Vector3 &v) const;
        Vector3 operator*(const double_sc &scalar) const;
        Vector3 operator/(const double_sc &scalar) const;
        Vector3 operator+=(const Vector3 &v);
        Vector3 operator-=(const Vector3 &v);
        Vector3 operator*=(const double_sc &scalar);
        Vector3 operator/=(const double_sc &scalar);
        Vector3 operator-() const;
        bool operator==(const Vector3 &v) const;
        bool operator!=(const Vector3 &v) const;
        double_sc operator[](const int index) const;
        double_sc operator[](const char index) const;

        double_sc dot(const Vector3 &v) const;
        Vector3 cross(const Vector3 &v) const;
        double_sc length() const;
        Vector3 normalize() const;
    };

}

#endif
