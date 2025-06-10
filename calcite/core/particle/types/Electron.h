//
// Created by jewel on 6/10/2025.
//

#ifndef ELECTRON_H
#define ELECTRON_H
#include "../Particle.h"
#include "../../math/ScaledDouble.h"

namespace particles {
    class Electron : public particles::Particle {
    public:
        [[nodiscard]] math::double_sc charge() const override;
        [[nodiscard]] math::double_sc spin() const override;
    };
}
#endif //ELECTRON_H
