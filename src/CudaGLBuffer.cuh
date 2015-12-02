#pragma once

#include "bothInclude.h"
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

class CudaGLBuffer
{
    cudaGraphicsResource *resource;
    GLuint buffer;
public:
    CudaGLBuffer();
    CudaGLBuffer(GLuint buf, unsigned int flags = cudaGraphicsMapFlagsWriteDiscard);
    void init(GLuint buf, unsigned int flags = cudaGraphicsMapFlagsWriteDiscard);
    void deinit();
    ~CudaGLBuffer ();
    bool mapResource();
    bool unmapResource();
    void *mappedPointer(size_t& numBytes) const;
    GLuint getId() const;
    cudaGraphicsResource *getResource() const;
};
