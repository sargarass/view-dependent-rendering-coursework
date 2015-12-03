#include "Application.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include "SystemManager.h"
#include "ConsoleWriter.h"

void Application::init() {
    m_frameIntervalValue = 0;
    pause = false;
    m_cameraMovable = true;
    m_fill = true;
    m_showDebug = true;
    m_mouse_capture = false;
    show_nurbs = true;
    m_running = false;


    m_triangles = 0;
    if (!m_killeroo.load("../share/models/killeroo.bmf")) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "", "", "model was not loaded");
        exit(-1);
    }

    if (!m_bigguy.load("../share/models/bigguy.bmf")) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "", "", "model was not loaded");
        exit(-1);
    }

    if (!m_teapot.load("../share/models/teapot.bmf")) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "", "", "model was not loaded");
        exit(-1);
    }

    glm::vec3 center = (1.0f / 6.0f) * (m_killeroo.box.min + m_killeroo.box.max + m_bigguy.box.min + m_bigguy.box.max + m_teapot.box.min + m_teapot.box.max);

    m_fps = 0;
    Window &window = SystemManager::getInstance()->window;
    window.open("Application", 1280, 700);
    window.mouse.setMousePosition(window.getWidth() / 2, window.getHeight() / 2);
    window.regApplication(this);


    glm::vec3 camEye(center.x, center.y, m_killeroo.box.min.z + (m_killeroo.box.min.z - m_killeroo.box.max.z) / 4);
    glm::vec3 camCenter(0, 0, center.z);
    glm::vec3 camUp( 0.0, 1.0, 0.0 );
    m_camera.translate(camEye);
    m_camera.lookAt(camCenter, camUp);


    glClearDepth(1.0f);
    glDepthFunc(GL_LESS);
    glShadeModel(GL_SMOOTH);

    SystemManager::getInstance()->stackAllocator.resize(512UL * 1024UL * 1024UL);
    SystemManager::getInstance()->vdRender.init(58UL * 1024UL * 1024UL);

    SystemManager::getInstance()->vdRender.loadPatches("killeroo", m_killeroo.getPatchesPtr(), m_killeroo.getSize());
    SystemManager::getInstance()->vdRender.loadPatches("bigguy", m_bigguy.getPatchesPtr(), m_bigguy.getSize());
    SystemManager::getInstance()->vdRender.loadPatches("teapot", m_teapot.getPatchesPtr(), m_teapot.getSize());
    SystemManager::getInstance()->window.setVSync(false);
}

void Application::save() {
    FILE* file = fopen("settings.bin", "wb");
    if (!file) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Application", "save", "file was not opened!");
        return;
    }
    fwrite(this, sizeof(Application), 1, file);
    fclose(file);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "save", "settings were saved!");
}

void Application::load() {
    FILE* file = fopen("settings.bin", "rb");
    if (!file) {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Application", "load", "file is not existing!");
        return;
    }

    fseek(file, 0L, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0L, SEEK_SET);

    if (size == sizeof(Application)) {
        char memory[sizeof(Application)];
        fread(memory, sizeof(Application), 1, file);
        Application *ptr = reinterpret_cast<Application*>(memory);
        m_running = ptr->m_running;
        m_camera = ptr->m_camera;
        pause = ptr->pause;
        m_cameraMovable = ptr->m_cameraMovable;
        m_fill = ptr->m_fill;

        Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "load", "settings were loaded %s", SourcePos());
    } else {
        Log::getInstance().write(LOG_MESSAGE_TYPE::ERROR, "Application", "load", "size != sizeof(Application) %s", SourcePos());

    }
    fclose(file);

}

void Application::deinit() {
    m_running = false;
    m_fps = 0;
}

void Application::render() {
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "Begin frame:");

    Window &window = SystemManager::getInstance()->window;
    glEnable(GL_DEPTH_TEST);
    if (!m_fill) {
        SystemManager::getInstance()->vdRender.setFill(VDFill::LINES);
    } else {
        SystemManager::getInstance()->vdRender.setFill(VDFill::FILL);
    }
    glDisable(GL_CULL_FACE);
    glEnable(GL_MULTISAMPLE);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    SystemManager::getInstance()->vdRender.beginFrame();

    glm::mat4 killerooModelMatrix = glm::mat4(1.0f);
    killerooModelMatrix = glm::scale(killerooModelMatrix, glm::vec3(2, 2, 2));
    //model = glm::rotate(model, 90.0f, glm::vec3(1, 0, 0));

    glm::mat4 view = m_camera.getViewMatrix();
    glm::mat4 perspective = m_camera.getProjectionMatrix();
    glm::mat4 PV = perspective * view;


    glm::mat4 MVP = PV * killerooModelMatrix;
    SystemManager::getInstance()->vdRender.setFrontFace(VDFrontFace::FRONT);
    SystemManager::getInstance()->vdRender.updateParameters(MVP, window.getWidth(), window.getHeight());

    SystemManager::getInstance()->vdRender.render("killeroo", 0.5);

    SystemManager::getInstance()->vdRender.setFrontFace(VDFrontFace::BACK);
    glm::mat4 bigguyModelMatrix = glm::mat4(1.0f);
    bigguyModelMatrix = glm::translate(bigguyModelMatrix, glm::vec3(m_killeroo.box.min.x - 200, m_killeroo.box.min.y, m_killeroo.box.min.z));
    bigguyModelMatrix = glm::rotate(bigguyModelMatrix, 180.0f, glm::vec3(0, 1, 0));
    bigguyModelMatrix = glm::scale(bigguyModelMatrix, glm::vec3(40, 40, 40));
    MVP = PV * bigguyModelMatrix;
    SystemManager::getInstance()->vdRender.updateParameters(MVP, window.getWidth(), window.getHeight());
    SystemManager::getInstance()->vdRender.render("bigguy", 0.5);

    SystemManager::getInstance()->vdRender.setFrontFace(VDFrontFace::NONE);
    glm::mat4 teapotModelMatrix = glm::mat4(1.0f);
    teapotModelMatrix = glm::translate(teapotModelMatrix, glm::vec3(m_killeroo.box.max.x + 200, m_killeroo.box.min.y, m_killeroo.box.min.z));
    teapotModelMatrix = glm::rotate(teapotModelMatrix, 90.0f, glm::vec3(0, 1, 0));
    teapotModelMatrix = glm::scale(teapotModelMatrix, glm::vec3(40, 40, 40));
    MVP = PV * teapotModelMatrix;
    SystemManager::getInstance()->vdRender.updateParameters(MVP, window.getWidth(), window.getHeight());
    SystemManager::getInstance()->vdRender.render("teapot", 0.5);
    SystemManager::getInstance()->vdRender.endFrame();

    VDRenderStatistics statistics = SystemManager::getInstance()->vdRender.getFrameStatistics();
    m_triangles += statistics.trianglesCount;
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "Frame render statistics:");
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "kernelMVP %f ms", statistics.kernelMVPNanoseconds / 1000000.0f);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "kernelOracle: %f ms", statistics.kernelOracleNanoseconds / 1000000.0f);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "kernelScan: %f ms", statistics.kernelScanNanoseconds / 1000000.0f);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "kernelSplit: %f ms", statistics.kernelSplitNanoseconds / 1000000.0f);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "glDraw: %f ms", statistics.glDrawNanoseconds / 1000000.0f);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "summary: %f ms", statistics.total / 1000000.0f);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "patches count: %zu", statistics.patchesCount);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "triangles count: %zu", statistics.trianglesCount);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "max memory queue size: %zu mb", statistics.maxMemoryQueueSizeInMB);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "max memory glBuffer size: %zu mb", statistics.maxMemoryGLBufferSizeInMB);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "max memory used queue: %zu mb", statistics.maxMemoryUsedQueueMB);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "max memory used glBuffer: %zu mb", statistics.maxMemoryUsedGLBufferInMB);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "glDrawCallsCounter: %zu ", statistics.drawCallsCounter);
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "render", "End frame:");
}

void Application::update() {
    KeyboardKeyRepeat();
}

void Application::run() {
    m_running = true;
    m_timer.start();
    Window &window = SystemManager::getInstance()->window;
    onWindowResize(window.getWidth(), window.getHeight());

    while(m_running) {
        m_frameIntervalTimer.start();

        if (!pause) {
            render();
            window.flush();
        }

        window.pullEvents();
        update();

        m_frameIntervalValue = m_frameIntervalTimer.elapsedNanoseconds();

        m_fps++;
        int time = static_cast<int>(m_timer.elapsedSeconds());
        if (time >= 2.0) {
            window.setTitle(std::string("Application: FPS = ") + std::to_string(m_fps / time) + std::string(" triangles per second ") + std::to_string((m_triangles / (double)(time)) / 1000000.0) + std::string("M"));
            m_fps = 0;
            m_triangles = 0;
            m_timer.start();
        }
    }

}

void Application::onMouseKeyPress(Mouse::Key key) {
    Log::getInstance().write(LOG_MESSAGE_TYPE::DEBUG, "Application", "onMouseKeyPress", "%s", Mouse::keyToString(key).c_str());
}

void Application::onKeyboardKeyRelease(Keyboard::Key key) {
    UNUSED_PARAM_HANDER(key);
}

void Application::KeyboardKeyRepeat() {
    Window &window = SystemManager::getInstance()->window;
    float cam_speed = 20.0f;
    if (window.keyboard.isKeyRepeated(Keyboard::W)) {
        if (m_cameraMovable) {
            m_camera.move(glm::vec3(0.0, 0.0, cam_speed), static_cast<float>(m_frameIntervalValue));
        }
    }

    if (window.keyboard.isKeyRepeated(Keyboard::S)) {
        if (m_cameraMovable) {
            m_camera.move(glm::vec3(0.0, 0.0, -cam_speed), static_cast<float>(m_frameIntervalValue));
        }
    }
    if (window.keyboard.isKeyRepeated(Keyboard::A)) {
        if (m_cameraMovable) {
            m_camera.move(glm::vec3(-cam_speed, 0.0, 0.0), static_cast<float>(m_frameIntervalValue));
        }
    }
    if (window.keyboard.isKeyRepeated(Keyboard::D)) {
        if (m_cameraMovable) {
            m_camera.move(glm::vec3(cam_speed, 0.0, 0.0), static_cast<float>(m_frameIntervalValue));
        }
    }
    if (window.keyboard.isKeyRepeated(Keyboard::PageUp)) {
        if (m_cameraMovable) {
            m_camera.move(glm::vec3(0.0, cam_speed, 0.0), static_cast<float>(m_frameIntervalValue));
        }
    }

    if (window.keyboard.isKeyRepeated(Keyboard::PageDown)) {
        if (m_cameraMovable) {
            m_camera.move(glm::vec3(0.0, -cam_speed, 0.0), static_cast<float>(m_frameIntervalValue));
        }
    }

}

void Application::onKeyboardKeyPress(Keyboard::Key key) {
    Window &window = SystemManager::getInstance()->window;

    switch (key) {
        case Keyboard::Escape:
            m_running = false;
            break;
        case Keyboard::Space:
            m_mouse_capture = !m_mouse_capture;
            window.showCursor(!m_mouse_capture);
            break;

        case Keyboard::F:
            m_fill = !m_fill;
            break;

        case Keyboard::B:
            save();
            break;

        case Keyboard::V:
            load();
            break;
        case Keyboard::P:
            pause = !pause;
            break;
        case Keyboard::F1:
            m_showDebug = !m_showDebug;
            ConsoleWriter::getInstance().showDebug(m_showDebug);
            break;
        default:
            break;
    }
}

void Application::onWindowResize(int width, int height) {
    glViewport(0, 0, width, height);
    m_camera.setProjection(45.0, (static_cast<double>(width))/(static_cast<double>(height)), 0.0001, 1000.0);
}

void Application::onMouseMove(int x, int y) {
    if (m_mouse_capture) {
        m_camera.setViewByMouse(x, y, static_cast<float>(m_frameIntervalValue));
    }
}
