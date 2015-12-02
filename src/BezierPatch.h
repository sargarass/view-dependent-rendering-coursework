#pragma once

#include "bothInclude.h"
#include <cuda_runtime.h>
#pragma pack(push, 1)
class BezierPatch {
public:
    __host__ __device__
    BezierPatch(){}

    __host__ __device__
    ~BezierPatch(){}

    __host__ __device__
    BezierPatch(BezierPatch const &tmp) {
        for (int i = 0; i < 16; i++) {
            row[i] = tmp.row[i];
        }
    }

    union {
        glm::vec4 points[4][4];
        glm::vec4 row[16];
        float rowf[64];
    };
};
#pragma pack(pop)

