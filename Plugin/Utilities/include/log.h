#pragma once
#include "IUnityGraphics.h"
#include "framework.h" 
#include <fstream>  
#include <string>
#include <vector>

extern "C"
{

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

    class UTILITIES_API Log
    {
    private:

        std::fstream _logFile;
        
        const size_t MAX_LOG_SIZE = 50;
        std::vector<std::string> _logInfoVector;
        std::vector<std::string> _logWarningVector;
        std::vector<std::string> _logErrorVector;
        std::string _logBufferSum;

    public:
        // SINGLETON
        static Log& log();

        Log(const Log&) = delete;
        Log& operator=(const Log&) = delete;
        Log(Log&&) = delete;
        Log& operator=(Log&&) = delete;


        Log();
        ~Log();

        /// <summary>
        /// It will merge all log and separate them by '\n' 
        /// </summary>
        /// <param name="data">a pointer to a string that will contains a 
        /// merge of all log and separate them by '\n'. We use a reference to 
        /// a const char* to permit marshalling</param>
        /// <param name="type">Type of log to get</param>
        void getMergedLog(const char*& data, logType type);

        /// <summary>
        /// Call this function to write an info in log
        /// </summary>
        /// <param name="text">info to write in log</param>
        void debugLog(const std::string text);

        /// <summary>
        /// Call this function to write an warning in log
        /// This will add Warning : before message
        /// </summary>
        /// <param name="text">warning to write in log</param>
        void debugLogWarning(const std::string text);


        /// <summary>
        /// Call this function to write an error in log
        /// This will add Error : before message
        /// </summary>
        /// <param name="text">error to write in log</param>
        void debugLogError(const std::string text);

        void extractLog(std::vector<std::string>& logVector);
    };

}
