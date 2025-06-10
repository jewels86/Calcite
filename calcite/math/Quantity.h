//
// Created by jewel on 6/10/2025.
//

#ifndef QUANTITY_H
#define QUANTITY_H
#include "ScaledDouble.h"
#include "Unit.h"

namespace math {
    template<typename U>
    class Quantity {
    public:
        math::double_sc value;

        constexpr Quantity();
        constexpr explicit Quantity(double value);
        constexpr explicit Quantity(const math::double_sc& value);

        constexpr Quantity(const Quantity<U>& other) = default;

        constexpr Quantity<U> operator+(const Quantity<U>& rhs) const;
        constexpr Quantity<U> operator-(const Quantity<U>& rhs) const;
        constexpr Quantity<U>& operator+=(const Quantity<U>& rhs);
        constexpr Quantity<U>& operator-=(const Quantity<U>& rhs);

        constexpr Quantity<U> operator*(double scalar) const;
        constexpr Quantity<U> operator/(double scalar) const;
        constexpr Quantity<U> operator*(const double_sc& scalar) const;
        constexpr Quantity<U> operator/(const double_sc& scalar) const;

        constexpr Quantity<U> operator*(const double_sc &scalar, const Quantity<U> &unit) {
            return Quantity<U>(scalar * unit.value);
        }
        constexpr Quantity<U> operator/(const double_sc &scalar, const Quantity<U> &unit) {
            return Quantity<U>(scalar / unit.value);
        }

        template<typename U2>
        constexpr auto operator*(const Quantity<U2>& rhs) const {
            using ResultUnit = Unit<
                U::length + U2::length,
                U::mass + U2::mass,
                U::time + U2::time,
                U::charge + U2::charge,
                U::temperature + U2::temperature,
                U::amount + U2::amount>;
            return Quantity<ResultUnit>(value * rhs.value);
        }

        template<typename U2>
        constexpr auto operator/(const Quantity<U2>& rhs) const {
            using ResultUnit = Unit<
                U::length - U2::length,
                U::mass - U2::mass,
                U::time - U2::time,
                U::charge - U2::charge,
                U::temperature - U2::temperature,
                U::amount - U2::amount>;
            return Quantity<ResultUnit>(value / rhs.value);
        }
    };

    template<typename U>
    class Quantity;

    // ScaledDouble * Quantity<U>
    template<typename U>
    constexpr Quantity<U> operator*(const ScaledDouble &lhs, const Quantity<U> &rhs) {
        return Quantity<U>(lhs * rhs.value);
    }

    // ScaledDouble / Quantity<U>
    template<typename U>
    constexpr Quantity<U> operator/(const ScaledDouble &lhs, const Quantity<U> &rhs) {
        return Quantity<U>(lhs / rhs.value);
    }

    // Quantity<U> * ScaledDouble
    template<typename U>
    constexpr Quantity<U> operator*(const Quantity<U> &lhs, const ScaledDouble &rhs) {
        return Quantity<U>(lhs.value * rhs);
    }

    // Quantity<U> / ScaledDouble
    template<typename U>
    constexpr Quantity<U> operator/(const Quantity<U> &lhs, const ScaledDouble &rhs) {
        return Quantity<U>(lhs.value / rhs);
    }

}



#endif //QUANTITY_H
