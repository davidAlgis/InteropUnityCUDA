#pragma once
#include "IUnityGraphics.h"
#include "framework.h" 
#include <fstream>  
#include <string>

class UTILITIES_API Log
{
    private:
        std::fstream m_logFile;
    public:
        // SINGLETON
        static Log& log();

        Log(const Log&) = delete;
        Log& operator=(const Log&) = delete;
        Log(Log&&) = delete;
        Log& operator=(Log&&) = delete;


        Log();
        ~Log();

        void debugLog(std::string text);
        
        void debugLogError(std::string text);
};
