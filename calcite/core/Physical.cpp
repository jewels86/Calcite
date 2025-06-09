//
// Created by jewel on 6/9/2025.
//

#include "Physical.h"

namespace core {
    math::Vector3 Physical::position(const int n) {
        return this->positions[n];
    }
    math::Vector3 Physical::velocity(const int n) {
        return this->velocities[n];
    }
}