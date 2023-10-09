#pragma once
#include "framework.h"
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

extern "C"
{

#define GRUMBLE(code, msg)                                                     \
    {                                                                          \
        if ((code) != 0)                                                       \
        {                                                                      \
            Log::log().debugLogError(msg);                                     \
            return code;                                                       \
        }                                                                      \
    }
#define SUCCESS_INTEROP_CODE 0

    enum class logType
    {
        Info,
        Warning,
        Error
    };

    /**
     * @brief       External function for dll, it will merge all log and
     * separate them by'\n'
     *
     * @param[in]  data     a pointer to a string that will contains a
     * merge of all log and separate them by '\n'. We use a reference to
     * a const char* to permit marshalling
     *
     * @param[in]  type     Type of log to get
     */
    void UTILITIES_API GetLastLog(const char *&data, logType type);

    class Log
    {
        private:
        std::fstream _logFile;

        const size_t MAX_LOG_SIZE = 50;
        std::vector<std::string> _logInfoVector;
        std::vector<std::string> _logWarningVector;
        std::vector<std::string> _logErrorVector;
        std::string _logBufferSum;
        std::function<void(std::string, std::fstream &)> _infoPrintFunction;
        std::function<void(std::string, std::fstream &)> _warningPrintFunction;
        std::function<void(std::string, std::fstream &)> _errorPrintFunction;

        public:
        // SINGLETON
        UTILITIES_API static Log &log();

        Log(const Log &) = delete;
        Log &operator=(const Log &) = delete;
        Log(Log &&) = delete;
        Log &operator=(Log &&) = delete;

        UTILITIES_API Log();
        UTILITIES_API ~Log();

        /**
         * @brief      This function is called when debulLog is called, it will
         * write the string in parameters and the fstream is the log file
         *
         * @param[in]  printFunc  The print function
         */
        UTILITIES_API void setInfoPrintFunction(
            std::function<void(std::string, std::fstream &)> printFunc);

        /**
         * @brief      This function is called when debulLogWarning is called,
         * it will write the string in parameters and the fstream is the log
         * file
         *
         * @param[in]  printFunc  The print function
         */
        UTILITIES_API void setWarningPrintFunction(
            std::function<void(std::string, std::fstream &)> printFunc);

        /**
         * @brief      This function is called when debulLogError is called, it
         * will write the string in parameters and the fstream is the log file
         *
         * @param[in]  printFunc  The print function
         */
        UTILITIES_API void setErrorPrintFunction(
            std::function<void(std::string, std::fstream &)> printFunc);

        /**
         * @brief       It will merge all log and separate them by '\n'.
         *
         * @param[in]  data  a pointer to a string that will contains a
         * merge of all log and separate them by '\n'. We use a reference to
         * a const char* to permit marshalling.
         *
         * param[in]  type  Type of log to get
         */
        UTILITIES_API void getMergedLog(const char *&data, logType type);

        /**
         * @brief      Call this function to write an info in log
         *
         * @param[in]  text  The text to write.
         */
        UTILITIES_API void debugLog(std::string text);

        /**
         * @brief      Call this function to write an warning in log
         * This will add Warning : before message
         *
         * @param[in]  text  warning to write in log
         */
        UTILITIES_API void debugLogWarning(std::string text);

        /**
         * @brief      Call this function to write an error in log
         * This will add Error : before message
         *
         * @param[in]  text  error to write in log
         */
        UTILITIES_API void debugLogError(std::string text);

        /**
         * @brief      Extract the last log of the log vector
         *
         * @param      logVector  The log vector
         */
        UTILITIES_API void extractLog(std::vector<std::string> &logVector);
    };
}
