//
// Created by jewel on 6/10/2025.
//

#include <stdexcept>
#include <cmath>

#include "ScaledDouble.h"

namespace math {
    ScaledDouble sqrt(const ScaledDouble &x) {
        if (x.mantissa < 0) {
            throw std::domain_error("Cannot compute square root of a negative number");
        }
        if (x.mantissa == 0) {
            return ScaledDouble(0.0);
        }
        double mantissa_sqrt = std::sqrt(x.mantissa);
        int exponent_sqrt = x.exponent / 2;
        if (x.exponent % 2 != 0) {
            mantissa_sqrt *= std::sqrt(2.0);
            exponent_sqrt += 1;
        }
        return ScaledDouble(mantissa_sqrt, exponent_sqrt);
    }
    inline ScaledDouble pow(const ScaledDouble &base, const ScaledDouble &exponent) {
        if (base.mantissa < 0 && exponent.mantissa != std::floor(exponent.mantissa)) {
            throw std::domain_error("Cannot compute power of a negative base with a non-integer exponent");
        }
        const double result_mantissa = std::pow(base.mantissa, exponent.mantissa);
        const int result_exponent = base.exponent * static_cast<int>(exponent.mantissa);
        return ScaledDouble(result_mantissa, result_exponent);
    }
    inline ScaledDouble log(const ScaledDouble &x) {
        if (x.mantissa <= 0) {
            throw std::domain_error("Cannot compute logarithm of a non-positive number");
        }
        const double mantissa_log = std::log(x.mantissa);
        const int exponent_log = x.exponent;
        return ScaledDouble(mantissa_log, exponent_log);
    }
    inline ScaledDouble exp(const ScaledDouble &x) {
        const double mantissa_exp = std::exp(x.mantissa);
        const int exponent_exp = x.exponent;
        return ScaledDouble(mantissa_exp, exponent_exp);
    }
    inline ScaledDouble sin(const ScaledDouble &x) {
        const double mantissa_sin = std::sin(x.mantissa);
        const int exponent_sin = x.exponent;
        return ScaledDouble(mantissa_sin, exponent_sin);
    }
    inline ScaledDouble cos(const ScaledDouble &x) {
        const double mantissa_cos = std::cos(x.mantissa);
        const int exponent_cos = x.exponent;
        return ScaledDouble(mantissa_cos, exponent_cos);
    }
    inline ScaledDouble tan(const ScaledDouble &x) {
        const double mantissa_tan = std::tan(x.mantissa);
        const int exponent_tan = x.exponent;
        return ScaledDouble(mantissa_tan, exponent_tan);
    }
}
