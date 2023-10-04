#include "log.h"
#include <ctime>
#include <utility>

#if !defined PROJECT_NAME
#define PROJECT_NAME ""
#endif

extern "C"
{

    void GetLastLog(const char *&data, logType type)
    {
        Log::log().getMergedLog(data, type);
    }

    Log &Log::log()
    {
        // Static are guarantee to be created only once and is thread safe
        // To be sure this is the only instance even across multiple processes
        // the declaration / initialisation must be declared in a compiled unit
        // ! This function should not be inline !
        static Log sInstance{};
        return sInstance;
    }

    Log::Log()
    {
        _logFile.open("log.txt", std::ios::out);

        std::string projectName = PROJECT_NAME;

        // get time now
        time_t now = time(NULL);
        tm now_tm = {};
        char str[26] = {};
        localtime_s(&now_tm, &now);
        asctime_s(str, 26, &now_tm);

        _logFile << "Log file - Project : " << projectName
                 << " - Begin at : " << str << std::endl;

        for (int i = 0; i < 128; i++)
        {
            _logFile << "-";
        }
        _logFile << std::endl;

        _logInfoVector.reserve(MAX_LOG_SIZE);
        _logWarningVector.reserve(MAX_LOG_SIZE);
        _logErrorVector.reserve(MAX_LOG_SIZE);
        _logBufferSum.reserve(128);
        _infoPrintFunction = [](const std::string &msg, std::fstream &logFile) {
            logFile << msg << std::endl;
            std::cout << msg << std::endl;
        };
        _warningPrintFunction = [](const std::string &msg,
                                   std::fstream &logFile) {
            logFile << msg << std::endl;
            std::cout << msg << std::endl;
        };

        _errorPrintFunction = [](const std::string &msg,
                                 std::fstream &logFile) {
            logFile << msg << std::endl;
            std::cout << msg << std::endl;
        };
    }

    Log::~Log()
    {
        std::string projectName = PROJECT_NAME;

        for (int i = 0; i < 128; i++)
        {
            _logFile << "-";
        }
        _logFile << std::endl;

        // get time now
        time_t now = time(nullptr);
        tm now_tm = {};
        char str[26] = {};
        localtime_s(&now_tm, &now);
        asctime_s(str, 26, &now_tm);

        _logFile << "End Log file - Project : " << projectName
                 << " - End at : " << str << std::endl;
        _logFile.close();
    }

    void Log::setInfoPrintFunction(
        std::function<void(std::string, std::fstream &)> printFunc)
    {
        _infoPrintFunction = std::move(printFunc);
    }

    void Log::setWarningPrintFunction(
        std::function<void(std::string, std::fstream &)> printFunc)
    {
        _warningPrintFunction = std::move(printFunc);
    }

    void Log::setErrorPrintFunction(
        std::function<void(std::string, std::fstream &)> printFunc)
    {
        _errorPrintFunction = std::move(printFunc);
    }

    void Log::debugLog(std::string text)
    {
        std::string msg = "Info " + text;
        _infoPrintFunction(text, _logFile);

        if (_logInfoVector.size() >= MAX_LOG_SIZE)
        {
            _logInfoVector.pop_back();
        }

        _logInfoVector.push_back(text);
    }

    void Log::debugLogWarning(std::string text)
    {
        std::string msg = "Warning " + text;
        _warningPrintFunction(msg, _logFile);

        if (_logWarningVector.size() >= MAX_LOG_SIZE)
        {
            _logWarningVector.pop_back();
        }

        _logWarningVector.push_back(text);
    }

    void Log::debugLogError(std::string text)
    {
        std::string msg = "Error " + text;
        _errorPrintFunction(msg, _logFile);
        if (_logErrorVector.size() >= MAX_LOG_SIZE)
        {
            _logErrorVector.pop_back();
        }

        _logErrorVector.push_back(text);
    }

    void Log::getMergedLog(const char *&data, logType logType)
    {
        switch (logType)
        {
        case logType::Info: {
            extractLog(_logInfoVector);
            break;
        }
        case logType::Warning: {
            extractLog(_logWarningVector);
            break;
        }
        case logType::Error: {
            extractLog(_logErrorVector);
            break;
        }
        default:
            debugLogError("Unknown log type");
            return;
        }

        data = _logBufferSum.c_str();
    }

    void Log::extractLog(std::vector<std::string> &logVector)
    {
        _logBufferSum.clear();
        _logBufferSum.reserve(128 * logVector.size());
        for (auto &&str : logVector)
        {
            _logBufferSum += str + "\n";
        }

        // we'll clear log
        logVector.clear();
    }
}
