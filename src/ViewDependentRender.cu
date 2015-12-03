#include "ViewDependentRender.h"
#include "LibResources.h"
#include <cub/cub.cuh>
#include "SystemManager.h"

#define DEGREE 4
#define NUM_COMP 4
#define NUM_POINTS (16)
#define HELP_ARRAYS_COUNT 6
#define NUM_THREADS 512
#define PATCHPERBLOCK NUM_THREADS / 16
#define SQR(x) ((x) * (x))
#define NUM_TRIANGLES_IN_PATCH (sizeof(Triangles) / sizeof(glm::vec4) / 3)
#define NUM_POINTS_IN_TRIANGLES (NUM_TRIANGLES_IN_PATCH * 3)
__constant__ float gSLU[4][4];
__constant__ float gSRU[4][4];
__constant__ float gSLV[4][4];
__constant__ float gSRV[4][4];

__constant__ float gMVP[4][4];
__constant__ float gIMVP[4][4];
__constant__ float gClipSpacePlanes[6][4];
__constant__ uint32_t gWidth;
__constant__ uint32_t gHeight;

__constant__ uint8_t triangleThreadPointChooser[NUM_POINTS_IN_TRIANGLES];
__constant__ uint8_t edgeThreadPointChooser[((DEGREE - 2) * 4) * 3];

#pragma pack(push, 1)
struct PatchPointer{
    union {
        struct {
            float *x;
            float *y;
            float *z;
            float *w;
        };
        float *p[4];
    };
};
#pragma pack(pop)


__device__ __inline__
float cuda_min(float a, float b) {
    return (a < b)? a : b;
}

__device__ __inline__
float cuda_max(float a, float b) {
    return (a < b)? b : a;
}

__device__ __inline__
void warpReduceMin(volatile float *memory) {
    int idx = threadIdx.y * 4 + threadIdx.x;
    if (idx < 8) {
        memory[idx] = cuda_min(memory[idx], memory[idx + 8]);
        memory[idx] = cuda_min(memory[idx], memory[idx + 4]);
        memory[idx] = cuda_min(memory[idx], memory[idx + 2]);
        memory[idx] = cuda_min(memory[idx], memory[idx + 1]);
    }
}

__device__ __inline__
void warpReduceMax(volatile float *memory) {
    int idx = threadIdx.y * 4 + threadIdx.x;
    if (idx < 8) {
        memory[idx] = cuda_max(memory[idx], memory[idx + 8]);
        memory[idx] = cuda_max(memory[idx], memory[idx + 4]);
        memory[idx] = cuda_max(memory[idx], memory[idx + 2]);
        memory[idx] = cuda_max(memory[idx], memory[idx + 1]);
    }
}

__device__ __inline__
void warpReduceOr(volatile uint32_t *memory) {
    int idx = threadIdx.y * 4 + threadIdx.x;
    if (idx < 8) {
        memory[idx] |= memory[idx + 8];
        memory[idx] |= memory[idx + 4];
        memory[idx] |= memory[idx + 2];
        memory[idx] |= memory[idx + 1];
    }
}

__device__ __inline__
void warpReduceAnd(volatile uint32_t *memory) {
    int idx = threadIdx.y * 4 + threadIdx.x;
    if (idx < 8) {
        memory[idx] &= memory[idx + 8];
        memory[idx] &= memory[idx + 4];
        memory[idx] &= memory[idx + 2];
        memory[idx] &= memory[idx + 1];
    }
}


__device__ __inline__
float clamp(float v, float min, float max) {
    return cuda_max(cuda_min(v, max), min);
}

__device__ __inline__
float linearInterpolation(float const &a, float const &b, float t) {
    return (1.0f - t) * a + t * b;
}

__device__ __inline__
glm::vec2 linearInterpolation2D(glm::vec2 const &a, glm::vec2 const b, float t) {
    glm::vec2 res;
    res.x = (1.0 - t) * a.x + t * b.x;
    res.y = (1.0 - t) * a.y + t * b.y;
    return res;
}

__device__ __inline__
glm::vec4 linearInterpolation4D(glm::vec4 const &a, glm::vec4 const b, float t) {
    glm::vec4 res;
    res.x = (1.0 - t) * a.x + t * b.x;
    res.y = (1.0 - t) * a.y + t * b.y;
    res.z = (1.0 - t) * a.z + t * b.z;
    res.w = (1.0 - t) * a.w + t * b.w;
    return res;
}


__device__ __inline__
glm::vec4 bilinearInterpolation4D(float u, float v, glm::vec4 &a, glm::vec4 &b, glm::vec4 &c, glm::vec4 &d) {
    glm::vec4 tmp = linearInterpolation4D(a, b, u);
    glm::vec4 tmp2 = linearInterpolation4D(c, d, u);
    return linearInterpolation4D(tmp, tmp2, v);
}

void VDRender::init(uint64_t gpuMemorySize) {
    /*if (gpuMemorySize < 128ull * 1024ull * 1024ull) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "ViewDependentRender", "init", "memory size < 128M invalid");
        exit(-1);
        return;
    }*/

    if (gpuAlignSize * 10 > gpuMemorySize) {
        return;
    }

    m_settings.faceMode = VDFrontFace::NONE;
    m_settings.fillMode = VDFill::FILL;

    GpuStackAllocator& allocator = SystemManager::getInstance()->gpuStackAllocator;
    uint64_t cudaObjects = (5 * sizeof(uint64_t) + sizeof(DecisionBits) + 2 * sizeof(float) * NUM_COMP * NUM_POINTS);
    uint64_t openGLObjects = sizeof(Triangles);
    m_settings.maxQueueSize = (gpuMemorySize - gpuAlignSize * 10) / (cudaObjects + openGLObjects);
    allocator.resize(cudaObjects * m_settings.maxQueueSize);
    m_statistics.maxMemoryQueueSizeInMB = CONVERT_BYTES_TO_MB(m_settings.maxQueueSize * cudaObjects);
    m_statistics.maxMemoryGLBufferSizeInMB = CONVERT_BYTES_TO_MB(m_settings.maxQueueSize * openGLObjects);
    m_glPart.objectsInVBO = 0;

    glGenVertexArrays(1, &m_glPart.vao);
    if (m_glPart.vao == 0) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "ViewDependentRender", "init", "glGenVertexArrays == NULL at %s", SourcePos());
        exit(-1);
    }

    glGenBuffers(1, &m_glPart.vbo);
    if (m_glPart.vbo == 0) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "ViewDependentRender", "init", "glGenBuffers == NULL at %s", SourcePos());
        exit(-1);
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_glPart.vbo);
    glBufferData(GL_ARRAY_BUFFER, m_settings.maxQueueSize * openGLObjects, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    m_glPart.buffer.init(m_glPart.vbo);
    printOpenGLError();
    if (!m_glPart.shader.load("../share/shaders/BezierPatchVertexShader.glsl",
                    "../share/shaders/BezierPatchFragmentShader.glsl")) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "ViewDependentRender", "init",
                                 "shader was not load");
        exit(-1);
    }


    float SLU[4][4] = {{1.0f, 1.0f/2.0f, 1.0f/4.0f, 1.0f / 8.0f},
                      {0.0f, 1.0f/2.0f, 1.0f/2.0f, 3.0f / 8.0f},
                      {0.0f, 0.0f,      1.0f/4.0f, 3.0f / 8.0f},
                      {0.0f, 0.0f,      0.0f,      1.0f / 8.0f}};

    float SRU[4][4] = {{1.0f / 8.0f, 0.0f,      0.0f,        0.0f},
                      {3.0f / 8.0f, 1.0f/4.0f, 0.0f,        0.0f},
                      {3.0f / 8.0f, 1.0f/2.0f, 1.0f / 2.0f, 0.0f},
                      {1.0f / 8.0f, 1.0f/4.0f, 1.0f / 2.0f, 1.0f}};

    float SLV[4][4] = {{1.0f,     0.0f,                      0,    },
                      {1.0f/2.0f, 1.0f/2.0f,     0,          0     },
                      {1.0f/4.0f, 1.0/2.0f,  1.0f/4.0f,     0.0f   },
                      {1.0f/8.0f, 3.0f/8.0f, 3.0f/8.0f, 1.0f / 8.0f}};

    float SRV[4][4] = {{1.0f/8.0f, 3.0f/8.0f, 3.0f/8.0f, 1.0/8.0f},
                       {0.0f, 1.0f/4.0f, 1.0f/2.0f, 1.0f/4.0f},
                      {0.0f, 0.0f, 1.0/2.0f, 1.0f/2.0f},
                      {0.0f, 0.0f, 0.0f, 1.0f}};

    uint8_t trianglePointChooser[18 * 3] = /*{0, 4, 1, 5, 2, 6, 4, 8, 5, 9, 6, 10, 8, 12, 9, 13, 10, 14,
                                   1, 1, 2, 2, 3, 3, 5, 5, 6, 6, 7, 7, 9, 9, 10, 10, 11, 11,
                                   4, 5, 5, 6, 6, 7, 8, 9, 9, 10, 10, 11, 12, 13, 13, 14, 14, 15}*/
                                   {0, 1, 4, 4, 1, 5, 1, 2, 5, 5, 2, 6, 2, 3, 6, 6, 3, 7, 4, 5, 8,
                                   8, 5, 9, 5, 6, 9, 9, 6, 10, 6, 7, 10, 10, 7, 11, 8, 9, 12, 12,
                                   9, 13, 9, 10, 13, 13, 10, 14, 10, 11, 14, 14, 11, 15};
    uint8_t edgePointChooser[8 * 3] = {1, 2, 4,  8,  7,  11, 13, 14,
                                       0, 0, 0,  0,  3,  3,  12, 12,
                                       3, 3, 12, 12, 15, 15, 15, 15};

    cudaMemcpyToSymbol(gSLU, SLU, sizeof(float) * 16);
    cudaMemcpyToSymbol(gSRU, SRU, sizeof(float) * 16);
    cudaMemcpyToSymbol(gSLV, SLV, sizeof(float) * 16);
    cudaMemcpyToSymbol(gSRV, SRV, sizeof(float) * 16);
    cudaMemcpyToSymbol(triangleThreadPointChooser, trianglePointChooser, sizeof(uint8_t) * 18 * 3);
    cudaMemcpyToSymbol(edgeThreadPointChooser, edgePointChooser, sizeof(uint8_t) * 8 * 3);
    glm::vec4 clipSpacePlanes[6] = {{1, 0, 0, 1.0000001},{-1, 0, 0, 1.0000001},{0, 1, 0, 1.0000001}, {0, -1, 0, 1.0000001}, {0, 0, 1, 1.0000001}, {0, 0, -1, 1.0000001}};
    cudaMemcpyToSymbol(gClipSpacePlanes, clipSpacePlanes, sizeof(glm::vec4) * 6);
}

static dim3 gridConfigure(uint64_t problemSize, dim3 block) {
    dim3 MaxGridDim = {(uint)LibResouces::getCudaProperties(0).maxGridDimensionSize[0], (uint)LibResouces::getCudaProperties(0).maxGridDimensionSize[1], (uint)LibResouces::getCudaProperties(0).maxGridDimensionSize[2]};
    dim3 gridDim = {1, 1, 1};
    uint64_t blockSize = block.x * block.y * block.z;
    // По z
    if (problemSize > MaxGridDim.y * MaxGridDim.x * blockSize) {
        gridDim.z = problemSize / MaxGridDim.x * MaxGridDim.y * blockSize;
        problemSize = problemSize % MaxGridDim.x * MaxGridDim.y * blockSize;
    }
    // По y
    if (problemSize > MaxGridDim.x * blockSize) {
        gridDim.y = problemSize / MaxGridDim.x * blockSize;
        problemSize = problemSize % MaxGridDim.x * blockSize;
    }

    gridDim.x = (problemSize + blockSize - 1) / blockSize;
    return gridDim;
}

static dim3 gridConfigureZ(uint64_t problemSize, dim3 block) {
    dim3 block_tmp = block;
    block_tmp.x = block_tmp.y = 1;
    return gridConfigure(problemSize, block_tmp);
}

VDRender::~VDRender() {
    deinit();
}

void VDRender::deinit() {
    if (m_glPart.vbo) {
        glDeleteBuffers(1, &m_glPart.vbo);
        m_glPart.vbo = 0;
    }
    if (m_glPart.vao) {
        glDeleteVertexArrays(1, &m_glPart.vao);
        m_glPart.vao = 0;
    }
    m_glPart.buffer.deinit();

    for (auto i = m_models.begin(); i != m_models.end(); i++) {
        if ((*i).second.x) {
            gpuFree((*i).second.x);
            (*i).second.x = 0;
        }
        if ((*i).second.y) {
            gpuFree((*i).second.y);
            (*i).second.y = 0;
        }
        if ((*i).second.z) {
            gpuFree((*i).second.z);
            (*i).second.z = 0;
        }
        if ((*i).second.w) {
            gpuFree((*i).second.w);
            (*i).second.w = 0;
        }
    }
}

void VDRender::beginFrame() {
    m_statistics.clear();
    m_glPart.objectsInVBO = 0;
}

void VDRender::endFrame() {
    flushGL();
}

void VDRender::updateParameters(glm::mat4 const &MVP, uint32_t const &width, uint32_t const &height) {
    glm::mat4 IMVP = glm::inverse(MVP);
    float tMPV[4][4];
    float tIMPV[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tMPV[j][i] = MVP[i][j];
            tIMPV[j][i] = IMVP[i][j];
        }
    }
    cudaMemcpyToSymbol(gMVP, tMPV, sizeof(float) * 16);
    cudaMemcpyToSymbol(gIMVP, tIMPV, sizeof(float) * 16);
    cudaMemcpyToSymbol(gWidth, &width, sizeof(uint32_t));
    cudaMemcpyToSymbol(gHeight, &height, sizeof(uint32_t));


}

void VDRender::setFill(VDFill fillMode) {
    m_settings.fillMode = fillMode;
}

void VDRender::setFrontFace(VDFrontFace face) {
    m_settings.faceMode = face;
}

void VDRender::drawGL(uint64_t size, uint64_t level) {
    GLuint query;
    GLuint64 elapsed_time;
    glGenQueries(1, &query);
    glBeginQuery(GL_TIME_ELAPSED, query);

    printOpenGLError();
    uint64_t patches = size;
    size = size * NUM_POINTS_IN_TRIANGLES;
    m_glPart.shader.bind();
    m_glPart.shader.setVal("color", glm::vec4(1.0f / level, 0.0, 0.0, 1.0f));
    glBindVertexArray(m_glPart.vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_glPart.vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, NUM_COMP, GL_FLOAT, GL_FALSE, 0, 0);

    if (m_settings.fillMode == VDFill::FILL) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else {
        if (m_settings.fillMode == VDFill::LINES) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
    }
    glDrawArrays(GL_TRIANGLES, 0, size);


    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glDrawArrays(GL_TRIANGLES, 0, size);
    glDisableVertexAttribArray(0);

    glBindVertexArray(0);
    m_glPart.shader.unbind();

    glEndQuery(GL_TIME_ELAPSED);

    int done = 0;
    while (!done) {
        glGetQueryObjectiv(query, GL_QUERY_RESULT_AVAILABLE, &done);
    }
    glGetQueryObjectui64v(query, GL_QUERY_RESULT, &elapsed_time);
    glDeleteQueries(1, &query);
    printOpenGLError();

    m_statistics.glDrawNanoseconds += elapsed_time;
    m_statistics.patchesCount += patches;
    m_statistics.trianglesCount += NUM_TRIANGLES_IN_PATCH * patches;
    m_statistics.drawCallsCounter++;
}

__device__ __inline__
uint64_t getGlobalIdx3DZ() {
    uint64_t blockId = blockIdx.x
                 + blockIdx.y * gridDim.x
                 + gridDim.x * gridDim.y * blockIdx.z;
    return blockId * blockDim.z + threadIdx.z;
}

__device__ __inline__
uint64_t getGlobalIdx3DZXY()
{
    uint64_t blockId = blockIdx.x
             + blockIdx.y * gridDim.x
             + gridDim.x * gridDim.y * blockIdx.z;
    return blockId * (blockDim.x * blockDim.y * blockDim.z)
              + (threadIdx.z * (blockDim.x * blockDim.y))
              + (threadIdx.y * blockDim.x)
              + threadIdx.x;
}

__device__ __inline__
uint64_t getIdx() {
    return threadIdx.y * DEGREE + threadIdx.x;
}

__global__
void kernelTransfer(RenderModel dest, BezierPatch* src, uint64_t size) {
    uint64_t patchId = getGlobalIdx3DZ();
    if (patchId >= size) {
        return;
    }
    uint64_t idx = getIdx();

    dest.x[patchId * NUM_POINTS + idx] = src[patchId].row[idx].x;
    dest.y[patchId * NUM_POINTS + idx] = src[patchId].row[idx].y;
    dest.z[patchId * NUM_POINTS + idx] = src[patchId].row[idx].z;
    dest.w[patchId * NUM_POINTS + idx] = src[patchId].row[idx].w;
}

void VDRender::runKernelTransfer(RenderModel &dest, BezierPatch* src, uint64_t size) {
    size_t patches = PATCHPERBLOCK;
    dim3 block = dim3(DEGREE, DEGREE, patches);
    dim3 gridDim = gridConfigureZ(size, block);
    kernelTransfer<<<gridDim, block>>>(dest, src, size);
    cudaCheckErrors("runKernelTransfer");
}

bool VDRender::loadPatches(std::string const modelName, BezierPatch const *ramPatches, uint64_t size) {
    if (this->m_models.find(modelName) != m_models.end()) {
        return false;
    }

    BezierPatch* patch_tmp = gpuMalloc<BezierPatch>(size);
    RenderModel model;
    model.x = gpuMalloc<float>(size * NUM_POINTS);
    model.y = gpuMalloc<float>(size * NUM_POINTS);
    model.z = gpuMalloc<float>(size * NUM_POINTS);
    model.w = gpuMalloc<float>(size * NUM_POINTS);
    model.size = size;

    if (patch_tmp == nullptr || model.w == nullptr || model.x == nullptr || model.y == nullptr || model.z == nullptr) {
        if (patch_tmp) {
            gpuFree(patch_tmp);
        }
        if (model.w) {
            gpuFree(model.w);
        }
        if (model.x) {
            gpuFree(model.x);
        }
        if (model.y) {
            gpuFree(model.y);
        }
        if (model.z) {
            gpuFree(model.z);
        }

        Log::getInstance().write(LOG_MESSAGE_TYPE::WARNING, "ViewDependentRender", "loadPatches", "Model %s was not load!", modelName.c_str());
        return false;
    }
    cudaMemcpy(patch_tmp, ramPatches, size * sizeof(BezierPatch), cudaMemcpyHostToDevice);
    runKernelTransfer(model, patch_tmp, size);
    m_models.insert(std::pair<std::string, RenderModel>(modelName, std::move(model)));
    gpuFree(patch_tmp);
    return true;
}

__device__ __inline__
float cudaDot4D(float const *a, float const *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

__device__ __inline__
float cudaDot4D(float const *a, glm::vec4 const &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

__device__ __inline__
float cudaDot4D(glm::vec4 const &a, glm::vec4 const &b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

__device__ __inline__
float cudaDot4D(glm::vec4 const &a, float const *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

__global__
void kernelMVP(GpuQueue queue, RenderModel model) {
    uint64_t patchId = getGlobalIdx3DZ();
    int idx = getIdx();

    if (patchId >= queue.size) {
        return;
    }

    glm::vec4 src;
    glm::vec4 dest;
    src.x = model.x[patchId * NUM_POINTS + idx];
    src.y = model.y[patchId * NUM_POINTS + idx];
    src.z = model.z[patchId * NUM_POINTS + idx];
    src.w = model.w[patchId * NUM_POINTS + idx];

    for (int i = 0; i < 4; i++) {
        dest[i] = cudaDot4D(gMVP[i], src);
    }

    queue.x.getPointer()[patchId * NUM_POINTS + idx] = dest.x;
    queue.y.getPointer()[patchId * NUM_POINTS + idx] = dest.y;
    queue.z.getPointer()[patchId * NUM_POINTS + idx] = dest.z;
    queue.w.getPointer()[patchId * NUM_POINTS + idx] = dest.w;
}


void VDRender::runKernelMVP(GpuQueue &queue, RenderModel const &model) {
    Timer time;
    time.start();

    size_t patches = PATCHPERBLOCK;
    dim3 block(DEGREE, DEGREE, patches);
    dim3 gridDim = gridConfigureZ(queue.size, block);
    kernelMVP <<<gridDim, block>>> (queue, model);
    cudaThreadSynchronize();

    uint64_t elapsed = time.elapsedNanosecondsU64();
    m_statistics.kernelMVPNanoseconds += elapsed;
    cudaCheckErrors("KernelMVP");
}

__device__
void generatePrimitives(float const *patches, Triangles &pointer) {
    int idx = getIdx();
    // быстрее на 0.06 ms
    float *out = reinterpret_cast<float*>(pointer.points);
    for (int i = threadIdx.y; i < NUM_POINTS_IN_TRIANGLES; i += NUM_COMP) {
        out[i * NUM_COMP + threadIdx.x] = patches[threadIdx.x * NUM_POINTS + triangleThreadPointChooser[i]];
    }
//    if (idx < 12) {
//        int offset = idx >> 2;
//        pointer.points[0 + offset * 6][0][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (0 + offset) * DEGREE + 1];
//        pointer.points[0 + offset * 6][1][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (1 + offset) * DEGREE + 0];
//        pointer.points[0 + offset * 6][2][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (0 + offset) * DEGREE + 0];

//        pointer.points[2 + offset * 6][0][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (0 + offset) * DEGREE + 1];
//        pointer.points[2 + offset * 6][1][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (0 + offset) * DEGREE + 2];
//        pointer.points[2 + offset * 6][2][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (1 + offset) * DEGREE + 1];

//        pointer.points[4 + offset * 6][0][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (0 + offset) * DEGREE + 2];
//        pointer.points[4 + offset * 6][1][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (0 + offset) * DEGREE + 3];
//        pointer.points[4 + offset * 6][2][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (1 + offset) * DEGREE + 2];

//        pointer.points[1 + offset * 6][0][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (0 + offset) * DEGREE + 1];
//        pointer.points[1 + offset * 6][1][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (1 + offset) * DEGREE + 1];
//        pointer.points[1 + offset * 6][2][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (1 + offset) * DEGREE + 0];

//        pointer.points[3 + offset * 6][0][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (0 + offset) * DEGREE + 2];
//        pointer.points[3 + offset * 6][1][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (1 + offset) * DEGREE + 2];
//        pointer.points[3 + offset * 6][2][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (1 + offset) * DEGREE + 1];

//        pointer.points[5 + offset * 6][0][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (0 + offset) * DEGREE + 3];
//        pointer.points[5 + offset * 6][1][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (1 + offset) * DEGREE + 3];
//        pointer.points[5 + offset * 6][2][threadIdx.x] = patches[threadIdx.x * NUM_POINTS + (1 + offset) * DEGREE + 2];
//    }

}

__global__
void kernelTriangleGenerate(GpuQueue queue, Triangles *pointer) {
    uint64_t patchId = getGlobalIdx3DZ();
    int idx = getIdx();
    if (patchId >= queue.size) {
        return;
    }

    __shared__ volatile float patches[PATCHPERBLOCK][NUM_COMP][NUM_POINTS];
    float volatile *shared_pointer = reinterpret_cast<float volatile*>(patches) + threadIdx.z * NUM_COMP * NUM_POINTS;
    shared_pointer[0 * NUM_POINTS + idx] = queue.x.getPointer()[patchId * NUM_POINTS + idx];
    shared_pointer[1 * NUM_POINTS + idx] = queue.y.getPointer()[patchId * NUM_POINTS + idx];
    shared_pointer[2 * NUM_POINTS + idx] = queue.z.getPointer()[patchId * NUM_POINTS + idx];
    shared_pointer[3 * NUM_POINTS + idx] = queue.w.getPointer()[patchId * NUM_POINTS + idx];
    generatePrimitives(const_cast<float*>(shared_pointer), pointer[patchId]);

}

void runTriangleGenerate(GpuQueue const &queue, Triangles *pointer) {
    Timer time;
    time.start();

    size_t patches = PATCHPERBLOCK;
    dim3 block(DEGREE, DEGREE, patches);
    dim3 gridDim = gridConfigureZ(queue.size, block);
    kernelTriangleGenerate <<<gridDim, block>>> (queue, pointer);
    cudaThreadSynchronize();

    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "ViewDependentRender", "runTriangleGenerate", "total time %f", time.elapsedNanoseconds());
    cudaCheckErrors("runTriangleGenerate");
}

__device__ __inline__
void makeEdgesLinear(float volatile *patch, DecisionBits const &decision) {
    for (int i = threadIdx.y; i < 8; i += 4) {
        float t = ((i & 1) + 1) / 3.0f;
        int8_t point_id = edgeThreadPointChooser[0 * 8 + i];
        int8_t next_point_id = edgeThreadPointChooser[0 * 8 + (i & 1)? i - 1 : i + 1];
        if (decision.get(point_id) && decision.get(next_point_id)) {
            patch[point_id + threadIdx.x * NUM_POINTS] = (1.0f - t) * patch[edgeThreadPointChooser[1 * 8 + i] + threadIdx.x * NUM_POINTS] +
                                                                  t * patch[edgeThreadPointChooser[2 * 8 + i] + threadIdx.x * NUM_POINTS];
        }
    }
}
#define down_left(p, comp)       (p)[0 * DEGREE + 0 + (comp) * NUM_POINTS]
#define down_right(p, comp)      (p)[0 * DEGREE + 3 + (comp) * NUM_POINTS]
#define up_left(p, comp)         (p)[3 * DEGREE + 0 + (comp) * NUM_POINTS]
#define up_right(p, comp)        (p)[3 * DEGREE + 3 + (comp) * NUM_POINTS]
#define pointComp(p, i, j, comp) (p)[(i) * DEGREE + (j) + (comp) * NUM_POINTS]

//#define linear(a, b, t) ((1.0f - (t)) * (a) + ((t) * (b)))
#define linear(a, b, t) (a + (b - a) * t)
__device__ __inline__
void makeEdgesLinear0(float volatile *patch, DecisionBits const &decision) {
    if (decision.get(0, 0) && decision.get(0, 1) && decision.get(0, 2) && decision.get(0, 3)) {
        pointComp(patch, 0, 1, 3) = linear(down_left(patch, 3), down_right(patch, 3), 1.0f / 3.0f);
        pointComp(patch, 0, 2, 3) = linear(down_left(patch, 3), down_right(patch, 3), 2.0f / 3.0f);;
        for (int i = 0; i < 3; i++) {
            pointComp(patch, 0, 1, i) = linear(down_left(patch, i), down_right(patch, i), 1.0f / 3.0f);
            pointComp(patch, 0, 2, i) = linear(down_left(patch, i), down_right(patch, i), 2.0f / 3.0f);
        }
    }

    if (decision.get(3, 0) && decision.get(3, 1) && decision.get(3, 2) && decision.get(3, 3)) {
        pointComp(patch, 3, 1, 3) = linear(up_left(patch, 3), up_right(patch, 3), 1.0f / 3.0f);
        pointComp(patch, 3, 2, 3) = linear(up_left(patch, 3), up_right(patch, 3), 2.0f / 3.0f);
        for (int i = 0; i < 3; i++) {
            pointComp(patch, 3, 1, i) = linear(up_left(patch, i), up_right(patch, i), 1.0f / 3.0f);
            pointComp(patch, 3, 2, i) = linear(up_left(patch, i), up_right(patch, i), 2.0f / 3.0f);
        }
    }

    if (decision.get(0, 0) && decision.get(1, 0) && decision.get(2, 0) && decision.get(3, 0)) {
        pointComp(patch, 1, 0, 3) = linear(down_left(patch, 3), up_left(patch, 3), 1.0f / 3.0f);
        pointComp(patch, 2, 0, 3) = linear(down_left(patch, 3), up_left(patch, 3), 2.0f / 3.0f);;
        for (int i = 0; i < 3; i++) {
            pointComp(patch, 1, 0, i) = linear(down_left(patch, i), up_left(patch, i), 1.0f / 3.0f);
            pointComp(patch, 2, 0, i) = linear(down_left(patch, i), up_left(patch, i), 2.0f / 3.0f);
        }
    }

    if (decision.get(0, 3) && decision.get(1, 3) && decision.get(2, 3) && decision.get(3, 3)) {
        pointComp(patch, 1, 3, 3) = linear(down_right(patch, 3), up_right(patch, 3), 1.0f / 3.0f);
        pointComp(patch, 2, 3, 3) = linear(down_right(patch, 3), up_right(patch, 3), 2.0f / 3.0f);;
        for (int i = 0; i < 3; i++) {
            pointComp(patch, 1, 3, i) = linear(down_right(patch, i), up_right(patch, i), 1.0f / 3.0f);
            pointComp(patch, 2, 3, i) = linear(down_right(patch, i), up_right(patch, i), 2.0f / 3.0f);
        }
    }
}

__device__ __inline__
void loadPoints(float volatile *dest1, float volatile *dest2, PatchPointer &patch, int const idx) {
    dest1[0 * NUM_POINTS + idx] = patch.x[idx];
    dest1[1 * NUM_POINTS + idx] = patch.y[idx];
    dest1[2 * NUM_POINTS + idx] = patch.z[idx];
    dest1[3 * NUM_POINTS + idx] = patch.w[idx];

    dest2[0 * NUM_POINTS + idx] = dest1[0 * NUM_POINTS + idx];
    dest2[1 * NUM_POINTS + idx] = dest1[1 * NUM_POINTS + idx];
    dest2[2 * NUM_POINTS + idx] = dest1[2 * NUM_POINTS + idx];
    dest2[3 * NUM_POINTS + idx] = dest1[3 * NUM_POINTS + idx];
}

__device__ __inline__
void subdivide4(float t, PatchPointer &out, float volatile *sharedA, float volatile *sharedB) {
    float LL[NUM_COMP];
    float LR[NUM_COMP];
    float RL[NUM_COMP];
    float RR[NUM_COMP];
    int i,j;

    for (i = 0; i < NUM_COMP; i++) {
        LL[i] = 0;
        LR[i] = 0;
        for (j = 0; j < DEGREE; j++) {
            LL[i] += sharedA[(threadIdx.y * DEGREE + j) + i * NUM_POINTS] * gSLU[j][threadIdx.x];
            LR[i] += sharedA[(threadIdx.y * DEGREE + j) + i * NUM_POINTS] * gSRU[j][threadIdx.x];
        }
        sharedA[(threadIdx.y * DEGREE + threadIdx.x) + i * NUM_POINTS] = LL[i];
        sharedB[(threadIdx.y * DEGREE + threadIdx.x) + i * NUM_POINTS] = LR[i];
    }

    for (i = 0; i < NUM_COMP; i++) {
        LL[i] = 0;
        LR[i] = 0;
        RL[i] = 0;
        RR[i] = 0;

        for (j = 0; j < DEGREE; j++) {
            LL[i] += gSLV[threadIdx.y][j] * sharedA[(j * DEGREE + threadIdx.x) + i * NUM_POINTS];
            LR[i] += gSLV[threadIdx.y][j] * sharedB[(j * DEGREE + threadIdx.x) + i * NUM_POINTS];
            RL[i] += gSRV[threadIdx.y][j] * sharedA[(j * DEGREE + threadIdx.x) + i * NUM_POINTS];
            RR[i] += gSRV[threadIdx.y][j] * sharedB[(j * DEGREE + threadIdx.x) + i * NUM_POINTS];
        }
    }


    for (i = 0; i < NUM_COMP; i++) {
        out.p[i][0 * NUM_POINTS + threadIdx.y * DEGREE + threadIdx.x] = LL[i];
        out.p[i][1 * NUM_POINTS + threadIdx.y * DEGREE + threadIdx.x] = LR[i];
        out.p[i][2 * NUM_POINTS + threadIdx.y * DEGREE + threadIdx.x] = RL[i];
        out.p[i][3 * NUM_POINTS + threadIdx.y * DEGREE + threadIdx.x] = RR[i];
    }
}

__global__
void kernelSplit(GpuQueue queue, GpuQueue newQueue, Triangles *primitives,
                 uint64_t const *todoExclusiveSum, uint64_t const *doneExclusiveSum, DecisionBits const *threadDecision)
{
    uint64_t patchId = getGlobalIdx3DZ();
    uint64_t idx = getIdx();

    if (patchId >= queue.size || threadDecision[patchId].isCULL()) {
        return;
    }
    __shared__ volatile float sharedMemory[PATCHPERBLOCK][2][NUM_COMP][NUM_POINTS];

    float volatile *patchA = reinterpret_cast<float volatile*>(sharedMemory) + threadIdx.z * 2 * NUM_POINTS * NUM_COMP;
    float volatile *patchB = patchA + NUM_POINTS * NUM_COMP;
    PatchPointer patch;
    uint64_t offset = patchId * NUM_POINTS;
    patch.x = queue.x.getPointer() + offset;
    patch.y = queue.y.getPointer() + offset;
    patch.z = queue.z.getPointer() + offset;
    patch.w = queue.w.getPointer() + offset;

    loadPoints(patchA, patchB, patch, idx);
    //makeEdgesLinear0(patchA, threadDecision[patchId]);
    makeEdgesLinear(patchA, threadDecision[patchId]);
    if (!threadDecision[patchId].isReady()) {
        offset = todoExclusiveSum[patchId] * NUM_POINTS;
        patch.x = newQueue.x.getPointer() + offset;
        patch.y = newQueue.y.getPointer() + offset;
        patch.z = newQueue.z.getPointer() + offset;
        patch.w = newQueue.w.getPointer() + offset;

        subdivide4(0.5, patch, patchA, patchB);
    } else {
        generatePrimitives(const_cast<float*>(patchA), primitives[doneExclusiveSum[patchId]]);
    }
}

__device__ __inline__
float cross_z(glm::vec3 &a, glm::vec3 &b) {
    a = glm::normalize(a);
    b = glm::normalize(b);
    return (a.x * b.y) - (a.y * b.x);
}

__device__ __inline__
float cross_z(glm::vec2 &a, glm::vec2 &b) {
    a = glm::normalize(a);
    b = glm::normalize(b);
    return (a.x * b.y) - (a.y * b.x);
}
__device__
float test(glm::vec2 &a, glm::vec2 &b, glm::vec2 &c) {
    return (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x);
}

__device__ __inline__
bool testBackfaceCullingSharedMemory(glm::vec4* corners, VDFrontFace const &face) {
    glm::vec2 a (corners[0].x / corners[0].w, corners[0].y / corners[0].w);
    glm::vec2 b (corners[1].x / corners[1].w, corners[1].y / corners[1].w);
    glm::vec2 c (corners[2].x / corners[2].w, corners[2].y / corners[2].w);
    glm::vec2 d (corners[3].x / corners[3].w, corners[3].y / corners[3].w);

    if (face == VDFrontFace::FRONT) {
        return (test(a,b,c) > 0 && test (c, b, d) > 0);
    }
    return (test(a,b,c) < 0 && test (c, b, d) < 0);

    /*glm::vec2 u = b - a;
    glm::vec2 v = c - a;
    float z = cross_z(u, v);
    bool res = z > 0;
    u = d - b;
    v = a - b;
    z = cross_z(u, v);
    res = res && z > 0;
    u = a - c;
    v = d - c;
    z = cross_z(u, v);
    res = res && z > 0;
    u = c - d;
    v = b - d;
    z = cross_z(u, v);
    res = res && z > 0;*/

    //return res;
}


__device__
bool boundingBox(PatchPointer const &patch, int idx) {
    glm::vec4 point;
    point.x = patch.x[idx];
    point.y = patch.y[idx];
    point.z = patch.z[idx];
    point.w = patch.w[idx];

    uint32_t test;
    for (int i = 0; i < 6; i++) {
        test = ((static_cast<uint32_t>(__ballot(cudaDot4D(point, gClipSpacePlanes[i]) >= 0.0f))) >> ((threadIdx.z & 1)? 16 : 0)) & 0xFFFF;
        if (test == 0) {
            return false;
        }
    }

    return true;
}


__device__ __inline__
int getCorner(int i) {
    return i * 3 + (i > 1) * 6;
}


__device__ __inline__
void loadCornersToSharedMemory(float volatile *memory, PatchPointer const &pointer) {
    memory[threadIdx.x * 4 + threadIdx.y] = pointer.p[threadIdx.y][getCorner(threadIdx.x)]; // транспонирование
}



__device__
DecisionBits approxQuad(PatchPointer &patch, float const &threshold, glm::vec4 *corners, int idx) {
    DecisionBits decision = { 0 };
    glm::vec2 test;
    float w;
    glm::vec4 interpolate = bilinearInterpolation4D(threadIdx.x * 1.0f / 3.0f, threadIdx.y * 1.0f / 3.0f, corners[0], corners[1], corners[2], corners[3]);

    w = patch.w[idx];
    test.x = patch.x[idx] / w;
    test.y = patch.y[idx] / w;

    test.x = (test.x * 0.5f + 0.5f) * gWidth;
    test.y = (test.y * 0.5f + 0.5f) * gHeight;

    interpolate.x = (interpolate.x / interpolate.w * 0.5f + 0.5f) * gWidth;
    interpolate.y = (interpolate.y / interpolate.w * 0.5f + 0.5f) * gHeight;

    bool lessThenTreshold = (SQR((test.x - interpolate.x)) + SQR((test.y - interpolate.y))) <= SQR(threshold);
    decision.arr = ((static_cast<uint32_t>(__ballot(lessThenTreshold))) >> ((threadIdx.z & 1)? 16 : 0)) & 0xFFFF;
    return decision.arr;
}



__global__
void kernelOracle(GpuQueue const queue, float const threshold,
                  uint64_t *todo, uint64_t *done, DecisionBits *threadDecision, VDFrontFace const face, bool forceDone)
{
    uint64_t patchId = getGlobalIdx3DZ();
    int idx = getIdx();
    if (patchId >= queue.size) {
        return;
    }

    __shared__ float memory[PATCHPERBLOCK][4][NUM_COMP];
    PatchPointer patch;
    DecisionBits decision = {0};
    patch.x = queue.x.getPointer() + patchId * NUM_POINTS;
    patch.y = queue.y.getPointer() + patchId * NUM_POINTS;
    patch.z = queue.z.getPointer() + patchId * NUM_POINTS;
    patch.w = queue.w.getPointer() + patchId * NUM_POINTS;

    float* halfWarpMemory = reinterpret_cast<float*>(memory) + threadIdx.z * 4 * NUM_COMP;
    bool inscreen = boundingBox(patch, idx);

    if (!inscreen) {
        decision.setCULL();
    } else {
        // считываем угловые точки
        loadCornersToSharedMemory(halfWarpMemory, patch);

        if (forceDone == false) {
            decision = approxQuad(patch, threshold, reinterpret_cast<glm::vec4*>(halfWarpMemory), idx);
        } else {
            decision = 0xFFFF;
        }

        if (idx == 0 && decision.isReady() && face != VDFrontFace::NONE) {
            bool backface = testBackfaceCullingSharedMemory(reinterpret_cast<glm::vec4*>(halfWarpMemory), face);
            if (backface) {
                decision.setCULL();
            }
        }
    }

    if (idx == 0) {
        todo[patchId] = (decision.isReady() || decision.isCULL()) ? 0 : 4;
        threadDecision[patchId] = decision;
        done[patchId] = decision.isReady();
    }
}


void VDRender::runKernelOracle(GpuQueue &queue,
                               float threshold,
                               GpuPointer<uint64_t> &todo,
                               GpuPointer<uint64_t> &done,
                               GpuPointer<DecisionBits> &threadDecision,
                               VDFrontFace const &face,
                               bool const forceDone) {
    size_t patches = PATCHPERBLOCK;
    dim3 block = dim3(DEGREE, DEGREE, patches);
    dim3 gridDim = gridConfigureZ(queue.size, block);
    Timer time;
    time.start();
    kernelOracle<<<gridDim, block>>>(queue,
                                     threshold,
                                     todo.getPointer(),
                                     done.getPointer(),
                                     threadDecision.getPointer(),
                                     face,
                                     forceDone);

    cudaThreadSynchronize();
    uint64_t elapsed = time.elapsedNanosecondsU64();
    m_statistics.kernelOracleNanoseconds += elapsed;
    cudaCheckErrors("KernelOracle");
}


bool VDRender::runKernelScan(GpuPointer<uint64_t> const &array, uint64_t const size, uint64_t &sum, GpuPointer<uint64_t> &exclusiveSum) {
    Timer time;
    time.start();

    size_t cub_tmp_memory_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_memory_size, array.getPointer(), exclusiveSum.getPointer(), size);

    GpuPointer<char> cub_tmp_memory = SystemManager::getInstance()->gpuStackAllocator.alloc<char>(cub_tmp_memory_size);

    if (cub_tmp_memory.getPointer() == nullptr) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "ViewDependentRender", "runKernelScan", "Allocation problem %s", SourcePos().c_str());
        return false;
    }

    cub::DeviceScan::ExclusiveSum(cub_tmp_memory.getPointer(), cub_tmp_memory_size, array.getPointer(), exclusiveSum.getPointer(), size);
    sum = 0;
    cudaMemcpy(&sum, exclusiveSum.getPointer() + (size - 1), sizeof(uint64_t), cudaMemcpyDeviceToHost);
    SystemManager::getInstance()->gpuStackAllocator.free(cub_tmp_memory);
    cudaThreadSynchronize();
    uint64_t elapsed = time.elapsedNanosecondsU64();
    m_statistics.kernelScanNanoseconds += elapsed;
    cudaCheckErrors("KernelScan");
    return true;
}

void VDRender::runKernelSplit(GpuQueue &queue,
                                     GpuQueue &newQueue,
                                     Triangles *triangles,
                                     GpuPointer<uint64_t> todoExclusiveSum,
                                     GpuPointer<uint64_t> doneExclusiveSum,
                                     GpuPointer<DecisionBits> threadDecisionBits) {
    Timer time;
    time.start();
    size_t patches = PATCHPERBLOCK;
    dim3 block = dim3(4, 4, patches);
    dim3 gridDim = gridConfigureZ(queue.size, block);
    kernelSplit<<<gridDim, block>>>(queue,
                                    newQueue,
                                    triangles,
                                    todoExclusiveSum.getPointer(),
                                    doneExclusiveSum.getPointer(),
                                    threadDecisionBits.getPointer());
    cudaThreadSynchronize();
    uint64_t elapsed = time.elapsedNanosecondsU64();
    m_statistics.kernelSplitNanoseconds += elapsed;
    cudaCheckErrors("KernelSplit");
}

void VDRender::flushGL() {
    if (m_glPart.objectsInVBO > 0) {
        drawGL(m_glPart.objectsInVBO, 1);
        m_statistics.maxMemoryUsedGLBufferInMB = std::max(CONVERT_BYTES_TO_MB(m_glPart.objectsInVBO * sizeof(Triangles)), m_statistics.maxMemoryUsedGLBufferInMB);
        m_glPart.objectsInVBO = 0;
    }
}

void VDRender::render(std::string const name, float threshold, int maxlevel) {
    Timer time;
    time.start();
    auto model_it = this->m_models.find(name);
    if (model_it == m_models.end()) {
        return;
    }
    RenderModel const &model = (*model_it).second;

    if (model.size >= m_settings.maxQueueSize) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "ViewDependentRender", "render", "model.size >= m_settings.maxQueueSize! %zu>=%zu!", model.size, m_settings.maxQueueSize);
        return;
    }
    GpuStackAllocator &allocator = SystemManager::getInstance()->gpuStackAllocator;
    GpuQueue queue[2];
    int first = 0;
    int second = 1;
    int level = 1;

    allocator.pushPosition();
    queue[0].x = allocator.alloc<float>(m_settings.maxQueueSize * NUM_POINTS);
    queue[0].y = allocator.alloc<float>(m_settings.maxQueueSize * NUM_POINTS);
    queue[0].z = allocator.alloc<float>(m_settings.maxQueueSize * NUM_POINTS);
    queue[0].w = allocator.alloc<float>(m_settings.maxQueueSize * NUM_POINTS);
    queue[1].x = allocator.alloc<float>(m_settings.maxQueueSize * NUM_POINTS);
    queue[1].y = allocator.alloc<float>(m_settings.maxQueueSize * NUM_POINTS);
    queue[1].z = allocator.alloc<float>(m_settings.maxQueueSize * NUM_POINTS);
    queue[1].w = allocator.alloc<float>(m_settings.maxQueueSize * NUM_POINTS);

    GpuPointer<uint64_t> todo = allocator.alloc<uint64_t>(m_settings.maxQueueSize + 1);
    GpuPointer<uint64_t> done = allocator.alloc<uint64_t>(m_settings.maxQueueSize + 1);
    GpuPointer<uint64_t> todoExclusiveSum = allocator.alloc<uint64_t>(m_settings.maxQueueSize + 1);
    GpuPointer<uint64_t> doneExclusiveSum = allocator.alloc<uint64_t>(m_settings.maxQueueSize + 1);
    GpuPointer<DecisionBits> threadDecision = allocator.alloc<DecisionBits>(m_settings.maxQueueSize + 1);

    if (!queue[0].x.getPointer() || !queue[0].y.getPointer() || !queue[0].z.getPointer() || !queue[0].w.getPointer() ||
        !queue[1].x.getPointer() || !queue[1].y.getPointer() || !queue[1].z.getPointer() || !queue[1].w.getPointer() ||
        !todo.getPointer() || !done.getPointer() || !threadDecision.getPointer()) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::WARNING, "ViewDependentRender", "render", "Allocation problem %s", SourcePos().c_str());
        allocator.popPosition();
        return;
    }

    queue[0].size = model.size;
    queue[1].size = model.size;
    m_glPart.buffer.mapResource();
    uint64_t maxQueueSize = 0;
    uint64_t todoSize = 0;
    uint64_t doneSize = 0;
    runKernelMVP(queue[first], model);

    bool forceDone = false;

    while (queue[first].size > 0) {
        maxQueueSize = std::max(maxQueueSize, queue[first].size);
        runKernelOracle(queue[first], threshold, todo, done, threadDecision, m_settings.faceMode, forceDone);
        todoSize = 0;
        doneSize = 0;

        if (!runKernelScan(todo, queue[first].size + 1, todoSize, todoExclusiveSum)) {
            Log::getInstance().write(LOG_MESSAGE_TYPE::WARNING, "ViewDependentRender", "render", "runKernelScan failed; break %s", SourcePos().c_str());
            break;
        }

        if (!runKernelScan(done, queue[first].size + 1, doneSize, doneExclusiveSum)) {
            Log::getInstance().write(LOG_MESSAGE_TYPE::WARNING, "ViewDependentRender", "render", "runKernelScan failed; break %s", SourcePos().c_str());
            break;
        }

        if ((todoSize >= m_settings.maxQueueSize || level == maxlevel) && forceDone == false) {
            Log::getInstance().write(LOG_MESSAGE_TYPE::WARNING, "ViewDependentRender", "render", "break %s", SourcePos().c_str());
            forceDone = true;
            continue;
        }


        size_t tmp;
        Triangles *primitivesPtr = reinterpret_cast<Triangles*>(m_glPart.buffer.mappedPointer(tmp));

        if ((m_glPart.objectsInVBO + doneSize) <= m_settings.maxQueueSize) {
            primitivesPtr += m_glPart.objectsInVBO;
            m_glPart.objectsInVBO += doneSize;
        } else {
            m_glPart.buffer.unmapResource();
            flushGL();
            m_glPart.buffer.mapResource();
            m_glPart.objectsInVBO = doneSize;
        }

        runKernelSplit(queue[first], queue[second], primitivesPtr, todoExclusiveSum, doneExclusiveSum, threadDecision);
//        m_glPart.buffer.unmapResource();
//        drawGL(doneSize, level);
//        m_glPart.buffer.mapResource();
        std::swap(first, second);
        queue[first].size = todoSize;
        level++;
    }

    m_statistics.maxMemoryUsedQueueMB = std::max(CONVERT_BYTES_TO_MB(maxQueueSize * sizeof(float) * NUM_POINTS * NUM_COMP * 2), m_statistics.maxMemoryUsedQueueMB);
    m_glPart.buffer.unmapResource();
    allocator.popPosition();
    m_statistics.total += time.elapsedNanosecondsU64();
}
