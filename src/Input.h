#pragma once
#include "inttypes.h"
#include <unordered_map>
#include "Timer.h"
#include <deque>
class Window;

typedef enum {
    Repeat,
    Press,
    Release
} PressType;

class Keyboard {
public:
    typedef enum {
        dammy,
        A,
        B,
        C,
        D,
        E,
        F,
        G,
        H,
        I,
        J,
        K,
        L,
        M,
        N,
        O,
        P,
        Q,
        R,
        S,
        T,
        U,
        V,
        W,
        X,
        Y,
        Z,
        Num0,
        Num1,
        Num2,
        Num3,
        Num4,
        Num5,
        Num6,
        Num7,
        Num8,
        Num9,
        Escape,
        LControl,
        LShift,
        LAlt,
        LSystem,
        RControl,
        RShift,
        RAlt,
        RSystem,
        Menu,
        LBracket,
        RBracket,
        SemiColon,
        Comma,
        Period,
        Quote,
        Slash,
        BackSlash,
        Tilde,
        Equal,
        Dash,
        Space,
        Return,
        Back,
        Tab,
        PageUp,
        PageDown,
        End,
        Home,
        Insert,
        Delete,
        Add,
        Subtract,
        Multiply,
        Divide,
        Left,
        Right,
        Up,
        Down,
        Numpad0,
        Numpad1,
        Numpad2,
        Numpad3,
        Numpad4,
        Numpad5,
        Numpad6,
        Numpad7,
        Numpad8,
        Numpad9,
        F1,
        F2,
        F3,
        F4,
        F5,
        F6,
        F7,
        F8,
        F9,
        F10,
        F11,
        F12,
        F13,
        F14,
        F15,
        Pause,
        KEYBOARD_SIZE
    } Key;

    void init(Window* window);
    void deinit();

    bool isKeyPressed(Keyboard::Key key);
    bool isKeyReleased(Keyboard::Key key);
    bool isKeyRepeated(Keyboard::Key key);

    static std::string keyToString(Keyboard::Key key);

    void setKey(Key key, PressType type);
private:
    Timer m_timer;
    Window *m_window;
    PressType m_keys[KEYBOARD_SIZE];
    bool m_down[KEYBOARD_SIZE];

    double m_timeResponse;
    std::deque<Key> m_queue;
};

class Mouse {
    friend class Window;
public:
        typedef enum {
            dammy,
            Left,
            Right,
            MOUSE_SIZE
        } Key;
        void init(Window* window);
        void deinit();
        int getX();
        int getY();
        bool isKeyPressed(Mouse::Key key);
        bool isKeyReleased(Mouse::Key key);
        bool isKeyRepeated(Mouse::Key key);
        static std::string keyToString(Mouse::Key key);
        void setMousePosition(int x, int y);
        void setKey(Mouse::Key key, PressType type);
private:
        Timer m_timer;
        double m_timeResponse;
        int m_x;
        int m_y;
        PressType m_keys[MOUSE_SIZE];
        Window *m_window;
        bool m_down[MOUSE_SIZE];
};

