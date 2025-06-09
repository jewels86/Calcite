//
// Created by jewel on 6/9/2025.
//

#ifndef PHYSICAL_H
#define PHYSICAL_H
#include <vector>

#include "../math/Vector3.h"

namespace core {
class Physical {
private:
    std::vector<math::Vector3> positions;
    std::vector<math::Vector3> velocities;
public:
    virtual ~Physical() = default;

    virtual math::Vector3 position(double t) = 0;
    virtual math::Vector3 velocity(double t) = 0;
    virtual math::Vector3 position(int n);
    virtual math::Vector3 velocity(int n);
};
}





#endif //PHYSICAL_H
