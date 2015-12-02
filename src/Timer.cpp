#include "Timer.h"

typedef std::chrono::duration<uint64_t, std::milli> u64_milliseconds;
typedef std::chrono::duration<uint64_t, std::nano> u64_nanoseconds;

Timer::Timer() {
}

void Timer::start() {
     this->m_start = std::chrono::high_resolution_clock::now();
}

double Timer::elapsedNanoseconds() {
    this->m_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(this->m_end - this->m_start).count() * 1e-9;
}

double Timer::elapsedSeconds() {
    this->m_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::seconds>(this->m_end - this->m_start).count();
}

double Timer::elapsedMicroseconds() {
    this->m_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(this->m_end - this->m_start).count() * 1e-6;
}

double Timer::elapsedMilliseconds() {
    this->m_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(this->m_end - this->m_start).count() * 1e-3;
}

uint64_t Timer::elapsedMillisecondsU64() {
    this->m_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<u64_milliseconds>(this->m_end - this->m_start).count();
}

uint64_t Timer::elapsedNanosecondsU64() {
    this->m_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<u64_nanoseconds>(this->m_end - this->m_start).count();
}

