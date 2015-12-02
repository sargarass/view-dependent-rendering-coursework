#pragma once
#include "bothInclude.h"
#include <stdint.h>

template <typename T>
class GpuPointer {
public:
    __host__ __device__
    GpuPointer() {
        m_size = 0;
        m_pointer = nullptr;
    }
    __host__ __device__
    GpuPointer(T* pointer, uint64_t objSize) {
        m_pointer = pointer;
        m_size = objSize;
    }
    __host__ __device__
    T* const &getPointer() const {
        return m_pointer;
    }
    __host__ __device__
    uint64_t getMemorySize() const {
        return m_size;
    }
    __host__ __device__
    void operator=(GpuPointer const &b) {
        this->m_pointer = b.m_pointer;
        this->m_size = b.m_size;
    }

    uint64_t m_size;
    T* m_pointer;
};

class GpuStackAllocator {
public:
    ~GpuStackAllocator();
    void resize(uint64_t size);

    template<typename T>
    GpuPointer<T> alloc(uint64_t count = 1) {
        uint64_t objSize = sizeof(T) * count;
        uint64_t offset = gpuAlignSize - 1;
        if (m_top + offset + objSize <= m_memory + m_size) {
            uintptr_t aligned = (m_top + offset) & ~(gpuAlignSize - 1); // выравняли указатель
            m_top = m_top + offset + objSize; // сдвинули указатель на top стека
            return GpuPointer<T>(reinterpret_cast<T*>(aligned), objSize + offset);
        }
        return GpuPointer<T>(nullptr, 0);
    }

    void clear();

    template<typename T>
    bool free(GpuPointer<T> handle) {
        if (handle.getPointer() == nullptr) {
            return false;
        }

        uint64_t size = handle.getMemorySize();
        uint64_t offset = gpuAlignSize - 1;
        uintptr_t aligned = (m_top - size + offset) & ~(gpuAlignSize - 1);
        if (aligned == reinterpret_cast<uintptr_t>( handle.getPointer() )) {
            m_top -= size;
            return true;
        }
        return false;
    }

    uint64_t availableMemory() {
        return m_memory + m_size - m_top;
    }


    void pushPosition();
    bool popPosition();
    void deinit();
private:
    std::deque<uintptr_t> m_position;
    uint64_t m_size;
    uintptr_t m_memory;
    uintptr_t m_top;
};
