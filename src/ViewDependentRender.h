#pragma once
#include "bothInclude.h"
#include "Shader.h"
#include "CudaGLBuffer.cuh"
#include "DecisionBits.h"
#include "BezierPatch.h"
#include "GpuStackAllocator.h"

#pragma pack(push, 1)
struct Triangles {
    glm::vec4 points[18][3];
};

struct GpuQueue {
    GpuPointer<float> x;
    GpuPointer<float> y;
    GpuPointer<float> z;
    GpuPointer<float> w;
    uint64_t size; // Размер в патчах
};

struct RenderModel {
    float *x;
    float *y;
    float *z;
    float *w;
    uint64_t size;
};
#pragma pack(pop)

struct VDRenderStatistics {
    uint64_t kernelMVPNanoseconds;
    uint64_t kernelOracleNanoseconds;
    uint64_t kernelScanNanoseconds;
    uint64_t kernelSplitNanoseconds;
    uint64_t glDrawNanoseconds;
    uint64_t patchesCountFinal;
    uint64_t patchesCountTotalProcessed;
    uint64_t trianglesCount;
    uint64_t total;
    uint64_t maxMemoryQueueSizeInMB;
    uint64_t maxMemoryGLBufferSizeInMB;
    uint64_t maxMemoryUsedQueueMB;
    uint64_t maxMemoryUsedGLBufferInMB;
    uint64_t drawCallsCounter;
    void clear() {
        patchesCountTotalProcessed = 0;
        kernelMVPNanoseconds = 0;
        kernelOracleNanoseconds = 0;
        kernelScanNanoseconds = 0;
        kernelSplitNanoseconds = 0;
        glDrawNanoseconds = 0;
        patchesCountFinal = 0;
        trianglesCount = 0;
        total = 0;
        maxMemoryUsedQueueMB = 0;
        maxMemoryUsedGLBufferInMB = 0;
        drawCallsCounter = 0;
    }

};

struct OpenGLValues {
    GLuint vbo;
    GLuint query;
    uint64_t objectsInVBO;
    GLuint vao;
    Shader shader;
    CudaGLBuffer buffer;
};

enum class VDFrontFace { FRONT, BACK, NONE };
enum class VDFill { LINES, FILL };

struct VDSettings {
    VDFrontFace faceMode;
    VDFill fillMode;
    uint64_t maxQueueSize;
};

class VDRender {
public:
    ~VDRender();
    void init(uint64_t gpuMemorySize);
    void deinit();

    bool loadPatches(std::string const modelName, BezierPatch const *ramPatches, uint64_t size);
    void setFill(VDFill fillMode);
    void setFrontFace(VDFrontFace face);
    void updateParameters(glm::mat4 const &MVP, uint32_t const &width, uint32_t const &height);
    void render(std::string const name, float threshold, int maxlevel = 0);
    void beginFrame();
    void endFrame();

    VDRenderStatistics getFrameStatistics() {
        return m_statistics;
    }

private:
    void runKernelMVP(GpuQueue &queue, RenderModel const &model);
    void runKernelTransfer(RenderModel &dest, BezierPatch* src, uint64_t size);
    void drawGL(size_t size, uint64_t level);
    void flushGL();
    void runKernelOracle(GpuQueue &queue,
                         float threshold,
                         GpuPointer<uint64_t> &todo,
                         GpuPointer<uint64_t> &done,
                         GpuPointer<DecisionBits> &threadDecision, VDFrontFace const &face, bool const forceDone);
    bool runKernelScan(GpuPointer<uint64_t> const &array, uint64_t const size, uint64_t &sum, GpuPointer<uint64_t> &exclusiveSum);

    void runKernelSplit(GpuQueue &queue,
                        GpuQueue &newQueue,
                        Triangles *triangles,
                        GpuPointer<uint64_t> todoExclusiveSum,
                        GpuPointer<uint64_t> doneExclusiveSum,
                        GpuPointer<DecisionBits> threadDecisionBits);
    VDSettings m_settings;
    VDRenderStatistics m_statistics;
    OpenGLValues m_glPart;
    std::unordered_map<std::string, RenderModel> m_models;
};
