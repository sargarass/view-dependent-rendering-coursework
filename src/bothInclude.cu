#include "bothInclude.h"
#include "SystemManager.h"

GpuStackAllocator::~GpuStackAllocator(){
    if (m_memory) {
        gpuFree(reinterpret_cast<void*>( m_memory ));
        m_memory = 0;
    }
}

void GpuStackAllocator::resize(uint64_t size) {
    if (m_memory != 0) {
        gpuFree(reinterpret_cast<void*>( m_memory ));
        m_memory = 0;
    }
    m_memory = reinterpret_cast<uintptr_t>( gpuMalloc<char>(size) );
    m_top = m_memory;
    m_size = size;
    m_position.clear();
}

void GpuStackAllocator::deinit() {
    if (m_memory) {
        gpuFree(reinterpret_cast<void*>( m_memory ));
        m_memory = 0;
    }
}

void GpuStackAllocator::pushPosition() {
    m_position.push_back(m_top);
}

bool GpuStackAllocator::popPosition() {
    if (!m_position.empty()) {
        m_top = m_position.back();
        m_position.pop_back();
        return true;
    }
    return false;
}

void GpuStackAllocator::clear() {
    m_top = m_memory;
    m_position.clear();
}

void *GpuMemoryAllocator::alloc(uint64_t size) {
    void *tmp;
    cudaMalloc(&tmp, size);
    cudaCheckErrors("GpuMemoryAllocator::alloc");

    uintptr_t mem = reinterpret_cast<uintptr_t>(tmp);
    m_memory.insert(mem);
    return tmp;
}

void GpuMemoryAllocator::free(void *ptr) {
    uintptr_t mem = reinterpret_cast<uintptr_t>(ptr);
    if (m_memory.find(mem) == m_memory.end()) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "GpuMemoryAllocator", "free", "double free gpu memory %zu", ptr);
        exit(-1);
    }

    m_memory.erase(mem);
    cudaFree(ptr);
    cudaCheckErrors("GpuMemoryAllocator::free");
}

void GpuMemoryAllocator::freeAll() {
    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO, "GpuMemoryAllocator", "freeAll", "");
    for (auto &memoryPtr : m_memory) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::INFO, "GpuMemoryAllocator", "freeAll", "free %zu", memoryPtr);
        cudaFree(reinterpret_cast<void*>(memoryPtr));
    }
    m_memory.clear();
}

void gpuFree(void *memory_to_free) {
    SystemManager::getInstance()->gpuMemoryManager.free(memory_to_free);
}

void *gpuMallocImp(size_t size) {
    return SystemManager::getInstance()->gpuMemoryManager.alloc(size);
}
