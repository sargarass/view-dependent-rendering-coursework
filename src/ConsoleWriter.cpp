#include "ConsoleWriter.h"

void ConsoleWriter::notify(Message const &msg) {
    std::string tmp;
    switch(msg.type) {
        case LOG_MESSAGE_TYPE::DEBUG:
            if (m_showDebug) {
                tmp = std::string(BOLDWHITE) + "[DEBUG] " + msg.text + RESET + "\n";
            } else {
                return;
            }
            break;
        case LOG_MESSAGE_TYPE::ERROR:
            tmp = std::string(BOLDRED) + "[ERROR] " + msg.text + RESET + "\n";
            break;
        case LOG_MESSAGE_TYPE::INFO:
            tmp = std::string(BOLDBLUE) + "[INFO] " + msg.text + RESET + "\n";
            break;
        case LOG_MESSAGE_TYPE::WARNING:
            tmp = std::string(BOLDYELLOW) + "[WARNING] " + msg.text + RESET + "\n";
            break;
        default:
            tmp = std::string(BOLDYELLOW) + "[UNKNOWN] " + msg.text + RESET + "\n";
            break;
    }
    fwrite( tmp.c_str(), tmp.size() + 1, 1, stdout);
}

void ConsoleWriter::showDebug(bool b) {
    m_showDebug = b;
}
