#include "Log.h"
#include "ConsoleWriter.h"
#include <stdarg.h>
#include "string.h"

void Log::write(LOG_MESSAGE_TYPE type, std::string const &className, std::string const &functionName, const char *format, ...) {
    Message msg = {"" , type};

    std::string tmp;
    tmp.resize(256);

    va_list arglist;
    va_start( arglist, format );
    vsnprintf(&tmp[0], 255, format, arglist);
    va_end( arglist );


    if (className != "") {
        msg.text += className + "::";
    }

    if (functionName != "") {
        msg.text += functionName + "(): ";
    }
    msg.text  += tmp;
    push(msg);
}

void Log::push(Message &msg) {
    m_queue.push_back(msg);

    if (m_queue.size() > 5) {
        m_queue.pop_front();
    }

    for (auto subscriber : m_subscribers) {
        subscriber->notify(msg);
    }
}

void Log::subscribe(ILogSubscriber *subscriber) {
    m_subscribers.push_back(subscriber);
}

Log &Log::getInstance()  {
    static Log gLog;
    static bool init = false;
    if (init == false) {
        gLog.subscribe(&ConsoleWriter::getInstance());
        init = true;
    }
    return gLog;
}
