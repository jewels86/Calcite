//
// Created by jewel on 6/9/2025.
//

#ifndef PARTICLE_H
#define PARTICLE_H
#include <optional>
#include <vector>

#include "../../math/Vector3.h"
#include "../Physical.h"

namespace core {

class Particle : public Physical {
private:
    std::vector<math::Vector3> positions;
    std::vector<math::Vector3> velocities;

public:
    virtual double mass() const = 0;
    virtual double charge() const = 0;
    virtual double spin() const = 0;

    math::Vector3 position(double t) override;
    math::Vector3 velocity(double t) override;
};

} // core

#endif //PARTICLE_H
