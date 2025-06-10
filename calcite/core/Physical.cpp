//
// Created by jewel on 6/9/2025.
//

#include "Physical.h"

#include <algorithm>

#include "World.h"

namespace core {
    math::Vector3 Physical::position(const int n) {
        return this->positions[n];
    }
    math::Vector3 Physical::velocity(const int n) {
        return this->velocities[n];
    }

    math::Vector3 Physical::position(const double t) {
        const int n0 = static_cast<int>(t / World::deltaTime);
        const int n1 = n0 + 1;
        if (n1 >= positions.size()) {
            return positions.back();
        }
        const double alpha = (t - n0 * World::deltaTime) / World::deltaTime;
        return positions[n0] * (1 - alpha) + positions[n1] * alpha;
    }
    math::Vector3 Physical::velocity(const double t) {
        const int n0 = static_cast<int>(t / World::deltaTime);
        const int n1 = n0 + 1;
        if (n1 >= velocities.size()) {
            return velocities.back();
        }
        const double alpha = (t - n0 * World::deltaTime) / World::deltaTime;
        return velocities[n0] * (1 - alpha) + velocities[n1] * alpha;
    }

    void Physical::addPosition(math::Vector3 p) {
        this->positions.push_back(p);
    }
    void Physical::addVelocity(math::Vector3 v) {
        this->velocities.push_back(v);
    }
    void Physical::removePosition(int n) {
        if (n >= 0 && n < positions.size()) {
            positions.erase(positions.begin() + n);
        }
    }
    void Physical::removeVelocity(int n) {
        if (n >= 0 && n < velocities.size()) {
            velocities.erase(velocities.begin() + n);
        }
    }
    void Physical::setPosition(math::Vector3 p, int n) {
        if (n >= 0 && n < positions.size()) {
            positions[n] = p;
        } else if (n == positions.size()) {
            addPosition(p);
        }
    }
    void Physical::setVelocity(math::Vector3 v, int n) {
        if (n >= 0 && n < velocities.size()) {
            velocities[n] = v;
        } else if (n == velocities.size()) {
            addVelocity(v);
        }
    }

    std::vector<Physical&> Physical::children() {
        return childrenList;
    }

    void Physical::addChild(Physical& child) {
        childrenList.push_back(child);
        child.parentPhysical = this;
    }

    void Physical::removeChild(Physical& child) {
        auto it = std::remove_if(childrenList.begin(), childrenList.end(),
            [&child](Physical& c) { return &c == &child; });
        if (it != childrenList.end()) {
            childrenList.erase(it, childrenList.end());
            child.parentPhysical = nullptr;
        }
    }

    std::optional<Physical&> Physical::parent() {
        if (parentPhysical) return *parentPhysical;
        return std::nullopt;
    }
}
