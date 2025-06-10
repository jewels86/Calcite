//
// Created by jewel on 6/9/2025.
//

#ifndef PHYSICAL_H
#define PHYSICAL_H
#include <optional>
#include <vector>

#include "../math/Vector3.h"

namespace core {
class Physical {
private:
    std::vector<math::Vector3> positions;
    std::vector<math::Vector3> velocities;
    std::vector<Physical&> childrenList;
    Physical* parentPhysical = nullptr;
public:
    virtual ~Physical() = default;

    virtual std<Physical&> children() = 0;
    virtual void addChild(Physical& child) = 0;
    virtual void removeChild(Physical& child) = 0;
    virtual std::optional<Physical&> parent() = 0;

    virtual double mass(int n) const = 0;
    virtual double mass(double t) const = 0;

    virtual math::Vector3 position(double t) = 0;
    virtual math::Vector3 velocity(double t) = 0;
    virtual math::Vector3 position(int n) = 0;
    virtual math::Vector3 velocity(int n) = 0;
    virtual void addPosition(math::Vector3 p) = 0;
    virtual void addVelocity(math::Vector3 v) = 0;
    virtual void removePosition(int n) = 0;
    virtual void removeVelocity(int n) = 0;
    virtual void setPosition(math::Vector3 p, int n) = 0;
    virtual void setVelocity(math::Vector3 v, int n) = 0;

};
}





#endif //PHYSICAL_H
