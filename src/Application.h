#pragma once
#include "Timer.h"
#include "LibResources.h"
#include "Input.h"
#include "Window.h"
#include "LibResources.h"
#include "Camera.h"
#include "Shader.h"
#include "ViewDependentRender.h"
#include "Nurbs.h"
#include "Мodel.h"

class Window;
class Application {
public:
    Application(){}
    ~Application(){}

    void init();
    void deinit();
    void run();

    void render();
    void update();

    void onKeyboardKeyPress(Keyboard::Key key);
    void onKeyboardKeyRelease(Keyboard::Key key);
    void KeyboardKeyRepeat();

    void onMouseKeyPress(Mouse::Key key);
    void onWindowResize(int width, int height);
    void onMouseMove(int x, int y);
    void save();
    void load();
private:
    bool show_nurbs;
    bool m_mouse_capture;
    bool m_running;
    bool m_showDebug;
    Timer  m_timer;
    Timer  m_frameIntervalTimer;
    double m_frameIntervalValue;
    int    m_fps;
    Camera m_camera;

    bool pause;
    bool m_cameraMovable;
    bool m_fill;
    Model m_killeroo;
    Model m_bigguy;
    Model m_teapot;
    uint64_t m_triangles;
    btree::btree_multiset<float> m_frameIntervalValues;
    uint64_t const m_maxFrameIntervalValuesSize = 100;
};
