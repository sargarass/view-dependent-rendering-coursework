#pragma once
#include <string>
#include "Input.h"
#include "Application.h"
class Application;

class Window {
    friend class Keyboard;
    friend class Mouse;
public:
    Window();
    ~Window();

    bool open(std::string window_name, int width, int height);
    void close();

    void setTitle(const std::string window_name);
    void setSize(int width, int height);

    void showCursor(bool show);
    int getWidth();
    int getHeight();
    std::string getTitle();

    void setActive();
    void setVSync(bool);
    void setFullScreen(bool fullscreen);

    bool isActive();
    bool isFullScreen();

    void flush();
    void pullEvents();

    Keyboard keyboard;
    Mouse    mouse;

    void regApplication(Application *app) {
        m_application = app;
    }
    void unregApplication() {
        m_application = nullptr;
    }

    bool isApplicationReg() {
        return m_application != nullptr;
    }

private:
    uintptr_t m_handle;
    int m_width;
    int m_height;
    Application *m_application;

    friend void keyCallback(void *, int key, int, int action, int);
    friend void mouseKeyCallback(void *, int button, int action, int);
    friend void resizeWindowCallback(void *, int, int);
    friend void cursorCallback(void *, double, double);
    void setMousePosition(int x, int y);
};
