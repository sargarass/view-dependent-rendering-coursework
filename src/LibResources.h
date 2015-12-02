#pragma once
#include "bothInclude.h"

struct CudaProperties {
    size_t maxThreadsPerBlock;
    size_t maxBlockDimensionSize[3];
    size_t maxGridDimensionSize[3];
    size_t totalGlobalMemorySizeBytes;
};

class LibResouces {
public:
    static void glewInit();
    static void glfwInit();

    static void deinit();

    static void cudaInit();


    static CudaProperties const &getCudaProperties(int device) {
        return m_cudaProperties[device];
    }

    static size_t getCudaDeviceCount() {
        return m_cudaDeviceCount;
    }
private:
    static void glfwDeinit();
    static bool m_glew_init;
    static bool m_glfw_init;
    static CudaProperties *m_cudaProperties;
    static size_t m_cudaDeviceCount;
};

