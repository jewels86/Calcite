//
// Created by jewel on 6/10/2025.
//

#include "ScaledDouble.h"
#include <cmath>

namespace math {
    void ScaledDouble::normalize() {
        if (mantissa == 0.0) {
            exponent = 0;
            return;
        }
        int exp;
        mantissa = std::frexp(mantissa, &exp);
        exponent += exp;
        // mantissa is now in [0.5, 1) or (-1, -0.5]
    }

    ScaledDouble::ScaledDouble(const double value) {
        if (value == 0.0) {
            mantissa = 0.0;
            exponent = 0;
        } else {
            int exp;
            mantissa = std::frexp(value, &exp);
            exponent = exp;
        }
    }

    ScaledDouble::ScaledDouble(const ScaledDouble &other)
        : mantissa(other.mantissa), exponent(other.exponent) {}
    ScaledDouble::ScaledDouble(const double mantissa, const int exponent) : mantissa(mantissa), exponent(exponent) {
        normalize();
    }

    ScaledDouble &ScaledDouble::operator=(const ScaledDouble &other) {
        if (this != &other) {
            mantissa = other.mantissa;
            exponent = other.exponent;
        }
        return *this;
    }

    ScaledDouble &ScaledDouble::operator=(const double &other) {
        if (other == 0.0) {
            mantissa = 0.0;
            exponent = 0;
        } else {
            int exp;
            mantissa = std::frexp(other, &exp);
            exponent = exp;
        }
        return *this;
    }

    ScaledDouble ScaledDouble::operator+(const ScaledDouble &other) const {
        if (mantissa == 0.0) return other;
        if (other.mantissa == 0.0) return *this;
        int exp_diff = exponent - other.exponent;
        double result_mantissa;
        int result_exponent;
        if (exp_diff >= 0) {
            result_mantissa = mantissa + std::ldexp(other.mantissa, -exp_diff);
            result_exponent = exponent;
        } else {
            result_mantissa = std::ldexp(mantissa, exp_diff) + other.mantissa;
            result_exponent = other.exponent;
        }
        ScaledDouble result;
        result.mantissa = result_mantissa;
        result.exponent = result_exponent;
        result.normalize();
        return result;
    }

    ScaledDouble ScaledDouble::operator-(const ScaledDouble &other) const {
        if (mantissa == 0.0) {
            ScaledDouble neg = other;
            neg.mantissa = -neg.mantissa;
            return neg;
        }
        if (other.mantissa == 0.0) return *this;
        const int exp_diff = exponent - other.exponent;
        double result_mantissa;
        int result_exponent;
        if (exp_diff >= 0) {
            result_mantissa = mantissa - std::ldexp(other.mantissa, -exp_diff);
            result_exponent = exponent;
        } else {
            result_mantissa = std::ldexp(mantissa, exp_diff) - other.mantissa;
            result_exponent = other.exponent;
        }
        ScaledDouble result;
        result.mantissa = result_mantissa;
        result.exponent = result_exponent;
        result.normalize();
        return result;
    }

    ScaledDouble ScaledDouble::operator*(const ScaledDouble &other) const {
        ScaledDouble result;
        result.mantissa = mantissa * other.mantissa;
        result.exponent = exponent + other.exponent;
        result.normalize();
        return result;
    }

    ScaledDouble ScaledDouble::operator/(const ScaledDouble &other) const {
        ScaledDouble result;
        result.mantissa = mantissa / other.mantissa;
        result.exponent = exponent - other.exponent;
        result.normalize();
        return result;
    }

    ScaledDouble &ScaledDouble::operator+=(const ScaledDouble &other) {
        *this = *this + other;
        return *this;
    }

    ScaledDouble &ScaledDouble::operator-=(const ScaledDouble &other) {
        *this = *this - other;
        return *this;
    }

    ScaledDouble &ScaledDouble::operator*=(const ScaledDouble &other) {
        *this = *this * other;
        return *this;
    }

    ScaledDouble &ScaledDouble::operator/=(const ScaledDouble &other) {
        *this = *this / other;
        return *this;
    }

    bool ScaledDouble::operator==(const ScaledDouble &other) const {
        if (mantissa == 0.0 && other.mantissa == 0.0) return true;
        return mantissa == other.mantissa && exponent == other.exponent;
    }

    bool ScaledDouble::operator!=(const ScaledDouble &other) const {
        return !(*this == other);
    }

    bool ScaledDouble::operator<(const ScaledDouble &other) const {
        if (mantissa == 0.0 && other.mantissa == 0.0) return false;
        if (exponent == other.exponent)
            return mantissa < other.mantissa;
        return exponent < other.exponent ? true : false;
    }

    bool ScaledDouble::operator<=(const ScaledDouble &other) const {
        return *this < other || *this == other;
    }

    bool ScaledDouble::operator>(const ScaledDouble &other) const {
        return !(*this <= other);
    }

    bool ScaledDouble::operator>=(const ScaledDouble &other) const {
        return !(*this < other);
    }

    // Double overloads

    ScaledDouble ScaledDouble::operator+(const double &other) const {
        return *this + ScaledDouble(other);
    }

    ScaledDouble ScaledDouble::operator-(const double &other) const {
        return *this - ScaledDouble(other);
    }

    ScaledDouble ScaledDouble::operator*(const double &other) const {
        return *this * ScaledDouble(other);
    }

    ScaledDouble ScaledDouble::operator/(const double &other) const {
        return *this / ScaledDouble(other);
    }

    ScaledDouble &ScaledDouble::operator+=(const double &other) {
        *this = *this + other;
        return *this;
    }

    ScaledDouble &ScaledDouble::operator-=(const double &other) {
        *this = *this - other;
        return *this;
    }

    ScaledDouble &ScaledDouble::operator*=(const double &other) {
        *this = *this * other;
        return *this;
    }

    ScaledDouble &ScaledDouble::operator/=(const double &other) {
        *this = *this / other;
        return *this;
    }

    bool ScaledDouble::operator==(const double &other) const {
        return *this == ScaledDouble(other);
    }

    bool ScaledDouble::operator!=(const double &other) const {
        return !(*this == other);
    }

    bool ScaledDouble::operator<(const double &other) const {
        return *this < ScaledDouble(other);
    }

    bool ScaledDouble::operator<=(const double &other) const {
        return *this <= ScaledDouble(other);
    }

    bool ScaledDouble::operator>(const double &other) const {
        return *this > ScaledDouble(other);
    }

    bool ScaledDouble::operator>=(const double &other) const {
        return *this >= ScaledDouble(other);
    }

    double ScaledDouble::toDouble() const {
        return std::ldexp(mantissa, exponent);
    }

    ScaledDouble ScaledDouble::operator-() const {
        ScaledDouble result;
        result.mantissa = -mantissa;
        result.exponent = exponent;
        return result;
    }

} // math