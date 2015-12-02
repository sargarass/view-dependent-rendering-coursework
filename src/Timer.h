#pragma once
#include "time.h"
#include <chrono>
#include "inttypes.h"

class Timer
{
public:
    Timer();
    void start();
    double elapsedNanoseconds();
    double elapsedSeconds();
    double elapsedMicroseconds();
    double elapsedMilliseconds();
    uint64_t elapsedMillisecondsU64();
    uint64_t elapsedNanosecondsU64();

private:
    std::chrono::high_resolution_clock::time_point  m_start;
    std::chrono::high_resolution_clock::time_point  m_end;
};


