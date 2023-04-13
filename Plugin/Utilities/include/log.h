#pragma once
#include "framework.h" 
#include <fstream>  
#include <vector>
#include <functional>
#include <string>
#include <iostream>

extern "C"
{

    #define GRUMBLE(code, msg) {if(code != 0){Log::log().debugLogError(msg); return code;}} 
    #define SUCCESS_INTEROP_CODE 0

    enum class logType
    {
        Info,
        Warning,
        Error
    };

    /// <summary>
    /// External function for dll, it will merge all log and separate them by '\n'
    /// </summary>
    /// <param name="data">a pointer to a string that will contains a 
    /// merge of all log and separate them by '\n'. We use a reference to 
    /// a const char* to permit marshalling</param>
    /// <param name="type">Type of log to get</param>
    void UTILITIES_API GetLastLog(const char*& data, logType type);

    class Log
    {
    private:

        std::fstream _logFile;
        
        const size_t MAX_LOG_SIZE = 50;
        std::vector<std::string> _logInfoVector;
        std::vector<std::string> _logWarningVector;
        std::vector<std::string> _logErrorVector;
        std::string _logBufferSum;
        std::function<void(std::string, std::fstream&)> _infoPrintFunction;
        std::function<void(std::string, std::fstream&)> _warningPrintFunction;
        std::function<void(std::string, std::fstream&)> _errorPrintFunction;

    public:
        // SINGLETON
        UTILITIES_API static Log& log();

        UTILITIES_API Log(const Log&) = delete;
        UTILITIES_API Log& operator=(const Log&) = delete;
        UTILITIES_API Log(Log&&) = delete;
        UTILITIES_API Log& operator=(Log&&) = delete;


        UTILITIES_API Log();
        UTILITIES_API ~Log();

        /// <summary>
        /// This function is called when debulLog is called, it will write the string in parameters 
        /// and the fstream is the log file 
        /// </summary>
        UTILITIES_API void setInfoPrintFunction(std::function<void(std::string, std::fstream&)> printFunc);


        /// <summary>
        /// This function is called when debulLogWarning is called, it will write the string in parameters 
        /// and the fstream is the log file 
        /// </summary>
        UTILITIES_API void setWarningPrintFunction(std::function<void(std::string, std::fstream&)> printFunc);


        /// <summary>
        /// This function is called when debulLogError is called, it will write the string in parameters 
        /// and the fstream is the log file 
        /// </summary>
        UTILITIES_API void setErrorPrintFunction(std::function<void(std::string, std::fstream&)> printFunc);

        /// <summary>
        /// It will merge all log and separate them by '\n' 
        /// </summary>
        /// <param name="data">a pointer to a string that will contains a 
        /// merge of all log and separate them by '\n'. We use a reference to 
        /// a const char* to permit marshalling</param>
        /// <param name="type">Type of log to get</param>
        UTILITIES_API void getMergedLog(const char*& data, logType type);

        /// <summary>
        /// Call this function to write an info in log
        /// </summary>
        /// <param name="text">info to write in log</param>
        UTILITIES_API void debugLog(const std::string text);

        /// <summary>
        /// Call this function to write an warning in log
        /// This will add Warning : before message
        /// </summary>
        /// <param name="text">warning to write in log</param>
        UTILITIES_API void debugLogWarning(const std::string text);


        /// <summary>
        /// Call this function to write an error in log
        /// This will add Error : before message
        /// </summary>
        /// <param name="text">error to write in log</param>
        UTILITIES_API void debugLogError(const std::string text);

        UTILITIES_API void extractLog(std::vector<std::string>& logVector);
    };

}
