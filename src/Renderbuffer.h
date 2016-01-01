#pragma once
#include "GL/glew.h"
#include "LibResources.h"
enum class AntialiasingAlgorithm {
    NONE,
    MSAA,
    CSAA
};

class Renderbuffer
{
public:
    void init();
    void deinit();
    void bind();
    void unbind();
    void flush();
    bool setAntialiasing(AntialiasingAlgorithm alg);
    bool setCSAAModes(uint mode);
    bool setMSAASamples(uint m_samples);
    uint getMSAAMaxSamples();
    std::vector<CSAAMode> const &getCSAAModes();
    bool isCSAASupported();
    AntialiasingAlgorithm getAntialiasingAlgorithm() { return m_algorithm; }
    void setWidth(uint width);
    void setHeight(uint height);
    uint getMSAASamples();
    uint getCSAAMode();
private:
    AntialiasingAlgorithm m_algorithm;
    struct {
        GLuint mode;
    } m_CSAA;
    GLuint m_width, m_height;
    GLuint m_samples;
    bool m_recalc;
    void changeBufferSettings();
    GLuint m_framebuffer;
    bool m_textureColorBufferInit;
    bool m_textureColorBufferMSInit;
    GLuint m_textureColorBuffer;
    GLuint m_textureColorBufferMS;

    GLuint m_renderBufferObject;
};
