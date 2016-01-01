#include "Renderbuffer.h"
#include "bothInclude.h"

void Renderbuffer::init() {
    m_framebuffer = 0;
    m_textureColorBuffer = 0;
    m_renderBufferObject = 0;
    m_samples = 1;
    m_textureColorBufferInit = m_textureColorBufferMS = 0;
    m_CSAA.mode = 0;
    m_algorithm = AntialiasingAlgorithm::NONE;
    m_width = m_height = 0;
    glGenRenderbuffers(1, &m_renderBufferObject);
    glGenFramebuffers(1, &m_framebuffer);
    glGenTextures(1, &m_textureColorBuffer);
    glGenTextures(1, &m_textureColorBufferMS);

    m_recalc = false;
}

void Renderbuffer::deinit() {
    if (m_renderBufferObject) {
        glDeleteRenderbuffers(1, &m_renderBufferObject);
    }
    if (m_framebuffer) {
        glDeleteFramebuffers(1, &m_framebuffer);
    }
    if (m_textureColorBuffer) {
        glDeleteTextures(1, &m_textureColorBuffer);
    }
    if (m_textureColorBufferMS) {
        glDeleteTextures(1, &m_textureColorBufferMS);
    }
}

void Renderbuffer::bind() {
    if (m_recalc) {
        changeBufferSettings();
        m_recalc = false;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer);
}

void Renderbuffer::unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Renderbuffer::changeBufferSettings() {
    switch (m_algorithm) {
        case AntialiasingAlgorithm::CSAA:
            glTextureImage2DMultisampleCoverageNV(m_textureColorBufferMS,
                                                  GL_TEXTURE_2D_MULTISAMPLE,
                                                  getCSAAModes()[m_CSAA.mode].coverageSamples,
                                                  getCSAAModes()[m_CSAA.mode].colorSamples,
                                                  GL_RGBA8,
                                                  m_width,
                                                  m_height,
                                                  GL_TRUE);
            glNamedFramebufferTexture(m_framebuffer, GL_COLOR_ATTACHMENT0, m_textureColorBufferMS, 0);
            glNamedRenderbufferStorageMultisampleCoverageEXT(m_renderBufferObject,
                                                       getCSAAModes()[m_CSAA.mode].coverageSamples,
                                                       getCSAAModes()[m_CSAA.mode].colorSamples,
                                                       GL_DEPTH24_STENCIL8,
                                                       m_width,
                                                       m_height);
            glNamedFramebufferRenderbuffer(m_framebuffer, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_renderBufferObject);
            if(glCheckNamedFramebufferStatus(m_framebuffer, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                    Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Application", "onWindowResize", "Framebuffer is not complete!");
                    exit(-1);
            }
            Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "onWindowResize", "Framebuffer is complete!");
            break;
        case AntialiasingAlgorithm::MSAA:
            glTextureImage2DMultisampleNV(m_textureColorBufferMS, GL_TEXTURE_2D_MULTISAMPLE, m_samples, GL_RGBA8, m_width, m_height, GL_TRUE);
            glNamedFramebufferTexture(m_framebuffer, GL_COLOR_ATTACHMENT0, m_textureColorBufferMS, 0);
            glNamedRenderbufferStorageMultisample(m_renderBufferObject, m_samples, GL_DEPTH24_STENCIL8, m_width, m_height);
            glNamedFramebufferRenderbuffer(m_framebuffer, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_renderBufferObject);
            if(glCheckNamedFramebufferStatus(m_framebuffer, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                    Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Application", "onWindowResize", "Framebuffer is not complete!");
                    exit(-1);
            }

            Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "onWindowResize", "Framebuffer is complete!");
            break;
       case AntialiasingAlgorithm::NONE:
            glTextureImage2DEXT(m_textureColorBuffer, GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            glTextureParameteriEXT(m_textureColorBuffer, GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteriEXT(m_textureColorBuffer, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glNamedFramebufferTextureEXT(m_framebuffer, GL_COLOR_ATTACHMENT0, m_textureColorBuffer, 0);
            glNamedRenderbufferStorageEXT(m_renderBufferObject, GL_DEPTH24_STENCIL8, m_width, m_height);
            glNamedFramebufferRenderbufferEXT(m_framebuffer, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_renderBufferObject);
            if(glCheckNamedFramebufferStatusEXT(m_framebuffer, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                    Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Application", "onWindowResize", "Framebuffer is not complete!");
                    exit(-1);
            }
            Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "onWindowResize", "Framebuffer is complete!");
            break;
    }
}

void Renderbuffer::setAntialiasing(AntialiasingAlgorithm alg) {
    m_algorithm = alg;
    m_recalc = true;
}

void Renderbuffer::setWidth(uint width) {
    m_width = width;
    m_recalc = true;
}

void Renderbuffer::setHeight(uint height) {
    m_height = height;
    m_recalc = true;
}

bool Renderbuffer::setCSAAModes(uint mode) {
    if (mode >= LibResouces::getCSAAProperties().modes.size()) {
        return false;
    }

    m_CSAA.mode = mode;
    if (m_algorithm == AntialiasingAlgorithm::CSAA) {
        m_recalc = true;
    }
    return true;
}

bool Renderbuffer::setMSAASamples(uint samples) {
    if (samples > LibResouces::getMSAAProperties().maxSamples || samples < 1) {
        return false;
    }

    m_samples = samples;
    if (m_algorithm == AntialiasingAlgorithm::MSAA) {
        m_recalc = true;
    }
    return true;
}

uint Renderbuffer::getMSAAMaxSamples() {
    return LibResouces::getMSAAProperties().maxSamples;
}

uint Renderbuffer::getMSAASamples() {
    return m_samples;
}

uint Renderbuffer::getCSAAMode() {
    return m_CSAA.mode;
}

std::vector<CSAAMode> const &Renderbuffer::getCSAAModes() {
    return LibResouces::getCSAAProperties().modes;
}

bool Renderbuffer::isCSAASupported() {
    return LibResouces::getCSAAProperties().isSupported;
}

void Renderbuffer::flush() {
    GLint dims[4];
    glGetIntegerv(GL_VIEWPORT, dims);
    glBlitNamedFramebuffer(m_framebuffer, 0, dims[0], dims[1], dims[2], dims[3], 0, 0, m_width, m_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}
