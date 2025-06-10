//
// Created by jewel on 6/9/2025.
//

#ifndef PARTICLE_H
#define PARTICLE_H
#include <optional>
#include <vector>

#include "../../math/Vector3.h"
#include "../Physical.h"

namespace particles {

class Particle : public Physical {
public:
    virtual ~Particle() = default;

    virtual double charge() const = 0;
    virtual double spin() const = 0;
};

} // core

#endif //PARTICLE_H
