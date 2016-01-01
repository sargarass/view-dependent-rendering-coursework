#include <unordered_map>
#include "Window.h"
#include "LibResources.h"
#include "GLFW/glfw3.h"
#include <iostream>
#include "SystemManager.h"

static Keyboard::Key glfw_keyboard_key_map[GLFW_KEY_LAST];
static Mouse::Key glfw_mouse_key_map[GLFW_MOUSE_BUTTON_LAST];
static PressType glfw_keyboard_key_press[3];

static void glfw_window_init() {
    static bool init = false;
    if (init) {
        return;
    }
    std::fill(glfw_keyboard_key_map, glfw_keyboard_key_map + GLFW_KEY_LAST, Keyboard::dammy);
    std::fill(glfw_mouse_key_map, glfw_mouse_key_map + GLFW_MOUSE_BUTTON_LAST, Mouse::dammy);
    std::fill(glfw_keyboard_key_press, glfw_keyboard_key_press + 3, PressType::Release);
    LibResouces::glfwInit();
    glfw_keyboard_key_map[GLFW_KEY_ESCAPE] = Keyboard::Escape;
    glfw_keyboard_key_map[GLFW_KEY_LEFT_ALT] = Keyboard::LAlt;
    glfw_keyboard_key_map[GLFW_KEY_LEFT_BRACKET] = Keyboard::LBracket;
    glfw_keyboard_key_map[GLFW_KEY_RIGHT_BRACKET] = Keyboard::RBracket;
    glfw_keyboard_key_map[GLFW_KEY_TAB] = Keyboard::Tab;
    glfw_keyboard_key_map[GLFW_KEY_LEFT_SHIFT] = Keyboard::LShift;
    glfw_keyboard_key_map[GLFW_KEY_RIGHT_SHIFT] = Keyboard::RShift;
    glfw_keyboard_key_map[GLFW_KEY_SPACE] = Keyboard::Space;
    glfw_keyboard_key_map[GLFW_KEY_LEFT] = Keyboard::Left;
    glfw_keyboard_key_map[GLFW_KEY_RIGHT] = Keyboard::Right;
    glfw_keyboard_key_map[GLFW_KEY_UP] = Keyboard::Up;
    glfw_keyboard_key_map[GLFW_KEY_DOWN] = Keyboard::Down;
    glfw_keyboard_key_map[GLFW_KEY_PAGE_UP] = Keyboard::PageUp;
    glfw_keyboard_key_map[GLFW_KEY_PAGE_DOWN] = Keyboard::PageDown;

    glfw_keyboard_key_map[GLFW_KEY_1] = Keyboard::Num1;
    glfw_keyboard_key_map[GLFW_KEY_2] = Keyboard::Num2;
    glfw_keyboard_key_map[GLFW_KEY_3] = Keyboard::Num3;
    glfw_keyboard_key_map[GLFW_KEY_4] = Keyboard::Num4;
    glfw_keyboard_key_map[GLFW_KEY_5] = Keyboard::Num5;
    glfw_keyboard_key_map[GLFW_KEY_6] = Keyboard::Num6;
    glfw_keyboard_key_map[GLFW_KEY_7] = Keyboard::Num7;
    glfw_keyboard_key_map[GLFW_KEY_8] = Keyboard::Num8;
    glfw_keyboard_key_map[GLFW_KEY_9] = Keyboard::Num9;

    glfw_keyboard_key_map[GLFW_KEY_KP_8] = Keyboard::Numpad8;
    glfw_keyboard_key_map[GLFW_KEY_KP_2] = Keyboard::Numpad2;
    glfw_keyboard_key_map[GLFW_KEY_KP_4] = Keyboard::Numpad4;
    glfw_keyboard_key_map[GLFW_KEY_KP_6] = Keyboard::Numpad6;
    glfw_keyboard_key_map[GLFW_KEY_F1] = Keyboard::F1;
    glfw_keyboard_key_map[GLFW_KEY_F2] = Keyboard::F2;
    glfw_keyboard_key_map[GLFW_KEY_F3] = Keyboard::F3;
    glfw_keyboard_key_map[GLFW_KEY_F4] = Keyboard::F4;
    glfw_keyboard_key_map[GLFW_KEY_F5] = Keyboard::F5;
    glfw_keyboard_key_map[GLFW_KEY_F6] = Keyboard::F6;
    glfw_keyboard_key_map[GLFW_KEY_F7] = Keyboard::F7;
    glfw_keyboard_key_map[GLFW_KEY_F8] = Keyboard::F8;
    glfw_keyboard_key_map[GLFW_KEY_F9] = Keyboard::F9;
    glfw_keyboard_key_map[GLFW_KEY_F10] = Keyboard::F10;
    glfw_keyboard_key_map[GLFW_KEY_F11] = Keyboard::F11;
    glfw_keyboard_key_map[GLFW_KEY_F12] = Keyboard::F12;
    glfw_keyboard_key_map[GLFW_KEY_F13] = Keyboard::F13;
    glfw_keyboard_key_map[GLFW_KEY_F14] = Keyboard::F14;
    glfw_keyboard_key_map[GLFW_KEY_F15] = Keyboard::F15;

    glfw_keyboard_key_map[GLFW_KEY_Q] = Keyboard::Q;
    glfw_keyboard_key_map[GLFW_KEY_W] = Keyboard::W;
    glfw_keyboard_key_map[GLFW_KEY_E] = Keyboard::E;
    glfw_keyboard_key_map[GLFW_KEY_R] = Keyboard::R;
    glfw_keyboard_key_map[GLFW_KEY_T] = Keyboard::T;
    glfw_keyboard_key_map[GLFW_KEY_Y] = Keyboard::Y;
    glfw_keyboard_key_map[GLFW_KEY_U] = Keyboard::U;
    glfw_keyboard_key_map[GLFW_KEY_I] = Keyboard::I;
    glfw_keyboard_key_map[GLFW_KEY_O] = Keyboard::O;
    glfw_keyboard_key_map[GLFW_KEY_P] = Keyboard::P;

    glfw_keyboard_key_map[GLFW_KEY_HOME] = Keyboard::Home;
    glfw_keyboard_key_map[GLFW_KEY_END] = Keyboard::End;



    glfw_keyboard_key_map[GLFW_KEY_A] = Keyboard::A;
    glfw_keyboard_key_map[GLFW_KEY_S] = Keyboard::S;
    glfw_keyboard_key_map[GLFW_KEY_D] = Keyboard::D;
    glfw_keyboard_key_map[GLFW_KEY_F] = Keyboard::F;
    glfw_keyboard_key_map[GLFW_KEY_G] = Keyboard::G;
    glfw_keyboard_key_map[GLFW_KEY_H] = Keyboard::H;
    glfw_keyboard_key_map[GLFW_KEY_J] = Keyboard::J;
    glfw_keyboard_key_map[GLFW_KEY_K] = Keyboard::K;
    glfw_keyboard_key_map[GLFW_KEY_L] = Keyboard::L;

    glfw_keyboard_key_map[GLFW_KEY_Z] = Keyboard::Z;
    glfw_keyboard_key_map[GLFW_KEY_X] = Keyboard::X;
    glfw_keyboard_key_map[GLFW_KEY_C] = Keyboard::C;
    glfw_keyboard_key_map[GLFW_KEY_V] = Keyboard::V;
    glfw_keyboard_key_map[GLFW_KEY_B] = Keyboard::B;
    glfw_keyboard_key_map[GLFW_KEY_N] = Keyboard::N;
    glfw_keyboard_key_map[GLFW_KEY_M] = Keyboard::M;
    glfw_mouse_key_map[GLFW_MOUSE_BUTTON_LEFT] = Mouse::Left;
    glfw_mouse_key_map[GLFW_MOUSE_BUTTON_RIGHT] = Mouse::Right;
    glfw_keyboard_key_press[GLFW_REPEAT] = PressType::Press;
    glfw_keyboard_key_press[GLFW_PRESS] = PressType::Press;
    glfw_keyboard_key_press[GLFW_RELEASE] = PressType::Release;
    init = true;
}

void keyCallback(void *, int key, int, int action, int) {
    Window &win = SystemManager::getInstance()->window;
    win.keyboard.setKey( glfw_keyboard_key_map[key], glfw_keyboard_key_press[action]);
    if (win.isApplicationReg()) {
        switch (action) {
            case GLFW_RELEASE:
                win.m_application->onKeyboardKeyRelease(glfw_keyboard_key_map[key]);
                break;
            case GLFW_PRESS:
                win.m_application->onKeyboardKeyPress(glfw_keyboard_key_map[key]);
                break;
        }
    }
}

void mouseKeyCallback(void *, int button, int action, int) {
    Window &win = SystemManager::getInstance()->window;
    win.mouse.setKey( glfw_mouse_key_map[button], glfw_keyboard_key_press[action]);
    if (win.isApplicationReg()) {
        win.m_application->onMouseKeyPress(glfw_mouse_key_map[button]);
    }
}

void resizeWindowCallback(void *, int width, int height) {
    Window &win = SystemManager::getInstance()->window;
    if (win.isApplicationReg()) {
        win.m_width = width;
        win.m_height = height;
        win.m_application->onWindowResize(width, height);
    }
}

void cursorCallback(void *, double x, double y) {
    Window &win = SystemManager::getInstance()->window;
    static int X = 0;
    static int Y = 0;

    if (win.isApplicationReg()) {
        win.m_application->onMouseMove(static_cast<int>(x - X), static_cast<int>(y - Y));
        X = x;
        Y = y;
    }
}

Window::Window() {
    m_handle = 0;
    m_application = nullptr;
    glfw_window_init();
}

Window::~Window() {
    close();
}

void Window::setMousePosition(int x, int y) {
    if (m_handle) {
        glfwSetCursorPos(reinterpret_cast<GLFWwindow*>(m_handle), x, y);
    }
}

void Window::setSize(int width, int height) {
    if (m_handle) {
        glfwSetWindowSize(reinterpret_cast<GLFWwindow*>(m_handle), width, height);
    }
}
bool Window::open(std::string window_name, int width, int height) {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if (m_handle) {
        close();
    }
    GLFWwindow* window = glfwCreateWindow(width, height, window_name.c_str(), NULL, NULL);
    if (window == nullptr) {
        return false;
    }
    m_height = height;
    m_width  = width;

    m_handle = reinterpret_cast<uintptr_t>(window);
    glfwMakeContextCurrent(window);
    auto ver_major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    auto ver_minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    auto profile = glfwGetWindowAttrib(window, GLFW_OPENGL_PROFILE);
    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO, "Window", "open", "GLFW_CONTEXT_VERSION_MAJOR %d", ver_major);
    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO, "Window", "open", "GLFW_CONTEXT_VERSION_MINOR %d", ver_minor);
    Log::getInstance().write(LOG_MESSAGE_TYPE::INFO, "Window", "open", "GLFW_OPENGL_PROFILE %d", profile);
    keyboard.init(this);
    mouse.init(this);
    glfwSetKeyCallback(window, reinterpret_cast<void(*)(GLFWwindow *, int key, int, int action, int)>(keyCallback));
    glfwSetWindowSizeCallback(window, reinterpret_cast<void(*)(GLFWwindow *, int width, int height)>(resizeWindowCallback));
    glfwSetMouseButtonCallback(window, reinterpret_cast<void(*)(GLFWwindow *, int button, int action, int)>(mouseKeyCallback));
    glfwSetCursorPosCallback(window, reinterpret_cast<void(*)(GLFWwindow *, double x, double y)>(cursorCallback));
    glfwSwapInterval(0);
    LibResouces::glewInit();


    return true;
}

void Window::setVSync(bool b) {
    glfwSwapInterval(b);
}

void Window::close() {
    if (m_handle) {
        keyboard.deinit();
        mouse.deinit();
        glfwDestroyWindow(reinterpret_cast<GLFWwindow*>(m_handle));
        m_handle = 0;
        m_application = nullptr;
        m_height = m_width = 0;
    }
}

void Window::flush() {
    if (m_handle) {
        glfwSwapBuffers(reinterpret_cast<GLFWwindow*>(m_handle));
    }
}

void Window::pullEvents() {
    glfwPollEvents();
}

void Window::showCursor(bool show) {
    if (m_handle) {
        if (show) {
            glfwSetInputMode(reinterpret_cast<GLFWwindow *>(m_handle), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }
        else {
            glfwSetInputMode(reinterpret_cast<GLFWwindow *>(m_handle), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        }
    }
}

int Window::getHeight() {
    if (m_handle) {
        return m_height;
    }
    return 0;
}

int Window::getWidth() {
    if (m_handle) {
        return m_width;
    }
    return 0;
}

void Window::setTitle(const std::string window_name) {
    if (m_handle) {
        glfwSetWindowTitle(reinterpret_cast<GLFWwindow*>(m_handle), window_name.c_str());
    }
}
