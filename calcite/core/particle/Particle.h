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

class Particle : public core::Physical {
public:
    [[nodiscard]] virtual math::double_sc charge() const = 0;
    [[nodiscard]] virtual math::double_sc spin() const = 0;
};

} // core

#endif //PARTICLE_H
