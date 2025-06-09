//
// Created by jewel on 6/9/2025.
//

#include "Particle.h"

namespace core {


    math::Vector3 Particle::position(const double t) {
        return this->positions[t / this->deltaTime];
    }

    math::Vector3 Particle::velocity(const double t) {
        return this->velocities[t / this->deltaTime];
    }

} // core