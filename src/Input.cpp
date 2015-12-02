#include "Input.h"
#include "string.h"
#include "Window.h"
#include <string>
void Keyboard::init(Window *window) {
    std::fill(m_keys, m_keys + Keyboard::KEYBOARD_SIZE, PressType::Release);
    std::fill(m_down, m_down + Keyboard::KEYBOARD_SIZE, false);
    m_window = window;
}

void Keyboard::deinit() {

}

bool Keyboard::isKeyPressed(Keyboard::Key key) {
    bool res = this->m_keys[key] == PressType::Press;

    if (res) {
        this->m_down[key] = false;
        this->m_keys[key] = PressType::Release;
    }

    return res;
}

bool Keyboard::isKeyReleased(Keyboard::Key key) {
    return (this->m_keys[key] == PressType::Release);
}

bool Keyboard::isKeyRepeated(Keyboard::Key key) {
    bool res = this->m_keys[key] == PressType::Repeat || this->m_keys[key] == PressType::Press;
    if (res && this->m_down[key] == false) {
        this->m_keys[key] = PressType::Release;
    }

    return res;
}

void Keyboard::setKey(Keyboard::Key key, PressType type) {
    if (type == PressType::Release) {
        m_down[key] = false;
        return;
    }

    m_down[key] = true;
    this->m_keys[key] = type;
}

std::string Keyboard::keyToString(Keyboard::Key key) {
    switch (key) {
        case A: return "Keyboard::A";
        case B: return "Keyboard::B";
        case C: return "Keyboard::C";
        case D: return "Keyboard::D";
        case E: return "Keyboard::E";
        case F: return "Keyboard::F";
        case G: return "Keyboard::G";
        case H: return "Keyboard::H";
        case I: return "Keyboard::I";
        case J: return "Keyboard::J";
        case K: return "Keyboard::K";
        case L: return "Keyboard::L";
        case M: return "Keyboard::M";
        case N: return "Keyboard::N";
        case O: return "Keyboard::O";
        case P: return "Keyboard::P";
        case Q: return "Keyboard::Q";
        case R: return "Keyboard::R";
        case S: return "Keyboard::S";
        case T: return "Keyboard::T";
        case U: return "Keyboard::U";
        case V: return "Keyboard::V";
        case W: return "Keyboard::W";
        case X: return "Keyboard::X";
        case Y: return "Keyboard::Y";
        case Z: return "Keyboard::Z";
        case Num0: return "Keyboard::Num0";
        case Num1: return "Keyboard::Num1";
        case Num2: return "Keyboard::Num2";
        case Num3: return "Keyboard::Num3";
        case Num4: return "Keyboard::Num4";
        case Num5: return "Keyboard::Num5";
        case Num6: return "Keyboard::Num6";
        case Num7: return "Keyboard::Num7";
        case Num8: return "Keyboard::Num8";
        case Num9: return "Keyboard::Num9";
        case Escape: return "Keyboard::Escape";
        case LControl: return "Keyboard::LControl";
        case LShift: return "Keyboard::LShift";
        case LAlt: return "Keyboard::LAlt";
        case LSystem: return "Keyboard::LSystem";
        case RControl: return "Keyboard::RControl";
        case RShift: return "Keyboard::RShift";
        case RAlt: return "Keyboard::RAlt";
        case RSystem: return "Keyboard::RSystem";
        case Menu: return "Keyboard::Menu";
        case LBracket: return "Keyboard::LBracket";
        case RBracket: return "Keyboard::RBracket";
        case SemiColon: return "Keyboard::SemiColon";
        case Comma: return "Keyboard::Comma";
        case Period: return "Keyboard::Period";
        case Quote: return "Keyboard::Quote";
        case Slash: return "Keyboard::Slash";
        case BackSlash: return "Keyboard::BackSlash";
        case Tilde: return "Keyboard::Tilde";
        case Equal: return "Keyboard::Equal";
        case Dash: return "Keyboard::Dash";
        case Space: return "Keyboard::Space";
        case Return: return "Keyboard::Return";
        case Back: return "Keyboard::Back";
        case Tab: return "Keyboard::Tab";
        case PageUp: return "Keyboard::PageUp";
        case PageDown: return "Keyboard::PageDown";
        case End: return "Keyboard::End";
        case Home: return "Keyboard::Home";
        case Insert: return "Keyboard::Insert";
        case Delete: return "Keyboard::Delete";
        case Add: return "Keyboard::Add";
        case Subtract: return "Keyboard::Subtract";
        case Multiply: return "Keyboard::Multiply";
        case Divide: return "Keyboard::Divide";
        case Left: return "Keyboard::Left";
        case Right: return "Keyboard::Right";
        case Up: return "Keyboard::Up";
        case Down: return "Keyboard::Down";
        case Numpad0: return "Keyboard::Numpad0";
        case Numpad1: return "Keyboard::Numpad1";
        case Numpad2: return "Keyboard::Numpad2";
        case Numpad3: return "Keyboard::Numpad3";
        case Numpad4: return "Keyboard::Numpad4";
        case Numpad5: return "Keyboard::Numpad5";
        case Numpad6: return "Keyboard::Numpad6";
        case Numpad7: return "Keyboard::Numpad7";
        case Numpad8: return "Keyboard::Numpad8";
        case Numpad9: return "Keyboard::Numpad9";
        case F1: return "Keyboard::F1";
        case F2: return "Keyboard::F2";
        case F3: return "Keyboard::F3";
        case F4: return "Keyboard::F4";
        case F5: return "Keyboard::F5";
        case F6: return "Keyboard::F6";
        case F7: return "Keyboard::F7";
        case F8: return "Keyboard::F8";
        case F9: return "Keyboard::F9";
        case F10: return "Keyboard::F10";
        case F11: return "Keyboard::F11";
        case F12: return "Keyboard::F12";
        case F13: return "Keyboard::F13";
        case F14: return "Keyboard::F14";
        case F15: return "Keyboard::F15";
        case Pause: return "Keyboard::Pause";
        default:
            break;
    }
    return "Keyboard::dammy";
}

void Mouse::init(Window *window) {
    std::fill(m_keys, m_keys + Mouse::MOUSE_SIZE, PressType::Release);
    m_window = window;
}

void Mouse::deinit() {

}

int Mouse::getX() {
    return m_x;
}

int Mouse::getY() {
    return m_y;
}

void Mouse::setKey(Mouse::Key key, PressType type) {
    m_keys[key] = type;
}

bool Mouse::isKeyPressed(Mouse::Key key) {
    return m_keys[key] == PressType::Press;
}

bool Mouse::isKeyReleased(Mouse::Key key) {
    return m_keys[key] == PressType::Release;
}

bool Mouse::isKeyRepeated(Mouse::Key key) {
    return m_keys[key] == PressType::Repeat;
}

void Mouse::setMousePosition(int x, int y) {
    if (m_window) {
        m_window->setMousePosition(x, y);
    }
}

std::string Mouse::keyToString(Mouse::Key key) {
    switch (key) {
        case Mouse::Key:: Left:
            return "Mouse::Left";
        case Mouse::Key::Right:
            return "Mouse::Right";
        default:
            break;
    }
    return "Mouse::dammy";
}
