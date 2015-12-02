#pragma once
#include "bothInclude.h"

struct AABB {
    glm::vec3 min, max;

    template<class T>
    void computeAABB(T* points, uint64_t size) {
        min = {FLT_MAX,FLT_MAX,FLT_MAX};
        max = {-FLT_MAX,-FLT_MAX,-FLT_MAX};
        for (uint64_t i = 0; i < size; i++) {
            min.x = std::min(min.x, points[i].x);
            min.y = std::min(min.y, points[i].y);
            min.z = std::min(min.z, points[i].z);

            max.x = std::max(max.x, points[i].x);
            max.y = std::max(max.y, points[i].y);
            max.z = std::max(max.z, points[i].z);
        }
    }
};
