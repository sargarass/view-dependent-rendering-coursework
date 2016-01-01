#pragma once
#include "bothInclude.h"

struct DecisionBits {
    int arr;

    __host__ __device__
    DecisionBits() { }

    __host__ __device__
    DecisionBits(uint16_t v) { arr = v; }

    __host__ __device__
    void set(uint at, uint b) {
        arr = arr | (b << at);
    }

    __host__ __device__
    void set(uint i, uint j, uint b) {
        arr = arr | (b << (i * 4 + j));
    }

    __host__ __device__
    bool get(uint i, uint j) const {
        return (arr & (1 << (i * 4 + j))) != 0;
    }

    __host__ __device__
    bool get(uint i) const {
        return (arr & (1 << (i))) != 0;
    }

    __host__ __device__
    void setCULL() {
        arr = 0;
    }

    __host__ __device__
    bool isCULL() const {
        return (arr == 0);
    }

    __host__ __device__
    bool isReady() const {
        return (arr == 0xFFFF);
    }
};
