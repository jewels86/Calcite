//
// Created by jewel on 6/10/2025.
//

#ifndef SCALEDDOUBLE_H
#define SCALEDDOUBLE_H

namespace math {

    class ScaledDouble {
    private:
        void normalize();

    public:
        double mantissa;
        int exponent;

        ScaledDouble(double value = 0.0);
        ScaledDouble(const ScaledDouble &other);
        ScaledDouble(double mantissa, int exponent);

        ScaledDouble &operator=(const ScaledDouble &other);
        ScaledDouble operator+(const ScaledDouble &other) const;
        ScaledDouble operator-(const ScaledDouble &other) const;
        ScaledDouble operator*(const ScaledDouble &other) const;
        ScaledDouble operator/(const ScaledDouble &other) const;
        ScaledDouble &operator+=(const ScaledDouble &other);
        ScaledDouble &operator-=(const ScaledDouble &other);
        ScaledDouble &operator*=(const ScaledDouble &other);
        ScaledDouble &operator/=(const ScaledDouble &other);
        bool operator==(const ScaledDouble &) const;
        bool operator!=(const ScaledDouble &) const;
        bool operator<(const ScaledDouble &) const;
        bool operator<=(const ScaledDouble &) const;
        bool operator>(const ScaledDouble &) const;
        bool operator>=(const ScaledDouble &) const;

        ScaledDouble &operator=(const double &other);
        ScaledDouble operator+(const double &other) const;
        ScaledDouble operator-(const double &other) const;
        ScaledDouble operator*(const double &other) const;
        ScaledDouble operator/(const double &other) const;
        ScaledDouble &operator+=(const double &other);
        ScaledDouble &operator-=(const double &other);
        ScaledDouble &operator*=(const double &other);
        ScaledDouble &operator/=(const double &other);
        bool operator==(const double &) const;
        bool operator!=(const double &) const;
        bool operator<(const double &) const;
        bool operator<=(const double &) const;
        bool operator>(const double &) const;
        bool operator>=(const double &) const;

        ScaledDouble operator-() const;

        double toDouble() const;
    };
    using double_sc = math::ScaledDouble;
} // math

#endif //SCALEDDOUBLE_H