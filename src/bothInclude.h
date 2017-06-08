#pragma once
#include <cinttypes>
#include <list>
#include "cpp-btree/btree_map.h"
#include "cpp-btree/btree_set.h"
#include <unordered_map>
#include <stack>
#include <deque>
#include <string>
#include <vector>
#include "Log.h"
#include "Timer.h"
#include <cuda_runtime.h>
#include <set>
#include <GL/glew.h>
#include <cmath>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>

#define CONVERT_BYTES_TO_MB(x) (((x) + 1024 * 1024 - 1) / (1024 * 1024))
const uint64_t cpuAlignSize = sizeof(int);
const uint64_t gpuAlignSize = 256;

inline void UNUSED_PARAM_HANDER(){}
template <typename Head, typename ...Tail>
inline void UNUSED_PARAM_HANDER(Head car, Tail ...cdr) { ((void) car); UNUSED_PARAM_HANDER(cdr...);}

inline void* operator new     ( size_t size) { return std::malloc( size ); }
inline void* operator new[]   ( size_t size) { return std::malloc( size ); }
inline void  operator delete  ( void* ptr) {
    std::free( ptr );
}
inline void  operator delete[]( void* ptr) {
    std::free( ptr );
}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "", "cudaCheckErrors", "%s (%s at %s)", \
                msg, cudaGetErrorString(__err), \
                SourcePos()); \
            \
            exit(-1); \
        } \
    } while (0)

#define printOpenGLError() { int line = __LINE__; const char* str = __FILE__;\
    GLenum glErr; \
    glErr = glGetError(); \
    if (glErr != GL_NO_ERROR) \
    {\
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "", "printOpenGLError", "%s (%s at %s)", \
                                 gluErrorString(glErr), str, line); \
        fflush(stdout); \
        exit(-1); \
    } \
}

class GpuMemoryAllocator {
public:
    void *alloc(uint64_t size);
    void free(void *ptr);
    void freeAll();
    ~GpuMemoryAllocator(){ freeAll(); }
private:
    btree::btree_set<uintptr_t> m_memory;
};


void* gpuMallocImp(size_t size);

template <typename T>
T* gpuMalloc(size_t size) {
    return static_cast<T*>(gpuMallocImp(sizeof(T) * size));
}

void gpuFree(void *memory_to_free);

