#include <iostream>
#include "LibResources.h"
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>
#include <cuda_runtime_api.h>

bool LibResouces::m_glew_init = false;
bool LibResouces::m_glfw_init = false;
CSAAProperties LibResouces::m_csaaProperties;
CudaProperties *LibResouces::m_cudaProperties = nullptr;
MSAAProperties LibResouces::m_msaaProperties;
size_t LibResouces::m_cudaDeviceCount = 0;

void error_callback(int, char const* description) {
    Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR,"", "error_callback",  "%s", description);
    exit(-1);
}

void LibResouces::glfwInit() {
    // GLFW initializing
    if (m_glfw_init) {
        return;
    }
    glfwSetErrorCallback(error_callback);
    if ( !::glfwInit() ) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR,"LibResouces", "glfwInit",  "GLFW was not initialized");
        exit(1);
    }

    m_glfw_init = true;
    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "glfwInit", "GLEW was initialized");
}

void LibResouces::glewInit() {
    //GLEW initializing
    if (m_glew_init) {
        return;
    }

    glewExperimental = GL_TRUE;
    GLenum err = ::glewInit();

    if (err != GLEW_OK) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR,"LibResouces", "glewInit", "GLEW was not initialized");
        exit(1);
    }
    m_glew_init = true;
    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "glewInit", "GLEW was initialized");

    GLenum glErr;
    do
    {
        glErr = glGetError();
    } while (glErr != GL_NO_ERROR);

    glGetIntegerv(GL_MAX_SAMPLES, &m_msaaProperties.maxSamples);

    // Поддержка CSAA
    m_csaaProperties.isSupported = glewIsSupported("GL_NV_framebuffer_multisample_coverage");
    if (m_csaaProperties.isSupported) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "glewInit", "CSAA is supported");
        Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "glewInit", "CSAA modes:");
        GLint numModes;
        glGetIntegerv(GL_MAX_MULTISAMPLE_COVERAGE_MODES_NV, &numModes);
        if (numModes)
        {
            GLint *modes = new GLint [2 * numModes];
            glGetIntegerv(GL_MULTISAMPLE_COVERAGE_MODES_NV, modes);
            m_csaaProperties.modes.resize(numModes);
            for (GLint i = 0; i < numModes; i++) {
                m_csaaProperties.modes[i].coverageSamples = modes[2 * i + 0];
                m_csaaProperties.modes[i].colorSamples = modes[2 * i + 1];
                Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "glewInit", "%d, %d", m_csaaProperties.modes[i].coverageSamples, m_csaaProperties.modes[i].colorSamples);
            }
            delete [] modes;
        }
    }
}

void LibResouces::glfwDeinit() {
    glfwTerminate();
    m_glfw_init = false;
}

void LibResouces::deinit() {
    glfwDeinit();
    if (m_cudaProperties) {
        delete [] m_cudaProperties;
    }
}

void LibResouces::cudaInit() {
    cudaDeviceReset();
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    m_cudaDeviceCount = deviceCount;
    if (deviceCount <= 0) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR,"LibResouces", "cudaInit", "No cuda compatible devices");
        exit(-1);
    }

    m_cudaProperties = new CudaProperties[deviceCount];
    for (int i = 0; i < deviceCount; i++) {
        int dev = i;
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "cudaInit", "Device %d: \"%s\"", dev, deviceProp.name);
        Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "cudaInit", "Compute capability %d.%d", deviceProp.major, deviceProp.minor);
        Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "cudaInit","Total amount of global memory: %.0f MBytes (%llu bytes)",
                                 static_cast<float>(deviceProp.totalGlobalMem)/1048576.0f, static_cast<unsigned long long>(deviceProp.totalGlobalMem));

        m_cudaProperties[i].totalGlobalMemorySizeBytes = deviceProp.totalGlobalMem;
        m_cudaProperties[i].maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

        for (int j = 0; j < 3; j++) {
            m_cudaProperties[i].maxBlockDimensionSize[j] = deviceProp.maxThreadsDim[j];
            m_cudaProperties[i].maxGridDimensionSize[j] = deviceProp.maxGridSize[j];
        }

        Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "cudaInit", "Max size of threads per block: %d", deviceProp.maxThreadsPerBlock);
        Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "cudaInit", "Max dimension size of a thread block (x,y,z): (%d, %d, %d)",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
       Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "cudaInit", "Max dimension size of a grid size    (x,y,z): (%d, %d, %d)",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
       cudaGLSetGLDevice(dev);
    }

    cudaCheckErrors("LibResouces:: cudaInit(): CUDA was not initialized");
    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO,"LibResouces", "cudaInit", "CUDA was initialized");
}
