//
// Created by jewel on 6/10/2025.
//

#include "Quantity.h"

namespace math {
    template<typename U>
    constexpr Quantity<U>::Quantity() : value(0.0) {}
    template<typename U>
    constexpr Quantity<U>::Quantity(const double value) : value(value) {}
    template<typename U>
    constexpr Quantity<U>::Quantity(const math::double_sc& value) : value(value) {}

    template<typename U>
    constexpr Quantity<U> Quantity<U>::operator+(const Quantity<U>& rhs) const {
        return Quantity<U>(value + rhs.value);
    }
    template<typename U>
    constexpr Quantity<U> Quantity<U>::operator-(const Quantity<U>& rhs) const {
        return Quantity<U>(value - rhs.value);

    }
    template<typename U>
    constexpr Quantity<U>& Quantity<U>::operator+=(const Quantity<U>& rhs) {
        value += rhs.value;
        return *this;
    }
    template<typename U>
    constexpr Quantity<U>& Quantity<U>::operator-=(const Quantity<U>& rhs) {
        value -= rhs.value;
        return *this;
    }

    template<typename U>
    constexpr Quantity<U> Quantity<U>::operator*(const double scalar) const {
        return Quantity<U>(value * scalar);
    }
    template<typename U>
    constexpr Quantity<U> Quantity<U>::operator/(const double scalar) const {
        return Quantity<U>(value / scalar);
    }
    template<typename U>
    constexpr Quantity<U> Quantity<U>::operator*(const double_sc& scalar) const {
        return Quantity<U>(value * scalar);
    }
    template<typename U>
    constexpr Quantity<U> Quantity<U>::operator/(const double_sc& scalar) const {
        return Quantity<U>(value / scalar);
    }
}