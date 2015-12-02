#pragma once
#include "bothInclude.h"

class StackAllocator {
public:
    StackAllocator(){
        m_size = 0;
        m_memory = 0;
        m_top = 0;
    }

    ~StackAllocator();

    template<typename T>
    T* alloc(uint64_t count = 1) {
        return reinterpret_cast<T*>( allocatorAlloc(sizeof(T) * count) );
    }

    void clear() {
        m_top = m_memory;
        m_position.clear();
    }

    bool free(void *pointer) {
        return allocatorFree(pointer);
    }

    void resize(uint64_t size);
    uint64_t availableMemory() {
        return m_memory + m_size - m_top;
    }

    void pushPosition();
    bool popPosition();
private:
    void *allocatorAlloc(uint64_t size);
    bool allocatorFree(void *pointer);

    uint64_t m_size;
    uintptr_t m_memory;
    uintptr_t m_top;
    std::deque<uintptr_t> m_position;
};
