#pragma once
#include "bothInclude.h"

struct CudaProperties {
    size_t maxThreadsPerBlock;
    size_t maxBlockDimensionSize[3];
    size_t maxGridDimensionSize[3];
    size_t totalGlobalMemorySizeBytes;
};

struct CSAAMode {
    GLint colorSamples;
    GLint coverageSamples;
};

struct CSAAProperties {
    bool isSupported;
    std::vector<CSAAMode> modes;
};

struct MSAAProperties {
    GLint maxSamples;
};

class LibResouces {
public:
    static void glewInit();
    static void glfwInit();
    static void deinit();
    static void cudaInit();

    inline static CudaProperties const &getCudaProperties(int device) {
        return m_cudaProperties[device];
    }

    inline static size_t getCudaDeviceCount() {
        return m_cudaDeviceCount;
    }

    inline static MSAAProperties const &getMSAAProperties() {
        return m_msaaProperties;
    }

    inline static CSAAProperties const &getCSAAProperties() {
        return m_csaaProperties;
    }

private:
    static void glfwDeinit();
    static bool m_glew_init;
    static bool m_glfw_init;
    static CudaProperties *m_cudaProperties;
    static CSAAProperties m_csaaProperties;
    static MSAAProperties m_msaaProperties;
    static size_t m_cudaDeviceCount;
};

