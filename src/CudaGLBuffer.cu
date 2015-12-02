#include "CudaGLBuffer.cuh"

CudaGLBuffer::CudaGLBuffer() {
    resource = nullptr;
    buffer = 0;
}

CudaGLBuffer::CudaGLBuffer (GLuint buf, unsigned int flags) {
    init(buf, flags);
}

void CudaGLBuffer::init(GLuint buf, unsigned int flags) {
    buffer = buf;
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
    cudaGraphicsGLRegisterBuffer ( &resource, buffer, flags );
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void CudaGLBuffer::deinit() {
    if (resource != nullptr) {
        cudaGraphicsUnregisterResource(resource);
        resource = nullptr;
        buffer = 0;
    }
}

CudaGLBuffer::~CudaGLBuffer() {
    deinit();
}

bool CudaGLBuffer::mapResource() {
    return cudaGraphicsMapResources (1, &resource) == cudaSuccess;
}

bool CudaGLBuffer::unmapResource() {
    return cudaGraphicsUnmapResources (1, &resource) == cudaSuccess;
}

void *CudaGLBuffer::mappedPointer(size_t& numBytes) const {
    void *ptr;
    if (cudaGraphicsResourceGetMappedPointer (&ptr, &numBytes, resource) != cudaSuccess)
        return NULL;
    return ptr;
}

GLuint CudaGLBuffer::getId() const {
    return buffer;
}

cudaGraphicsResource *CudaGLBuffer::getResource() const {
    return resource;
}
