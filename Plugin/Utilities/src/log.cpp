#include "log.h"
#include <ctime>

#if !defined PROJECT_NAME
#define PROJECT_NAME ""
#endif


Log& Log::log()
{
    // Static are guarantee to be created only once and is thread safe
    // To be sure this is the only instance even across multiple processes
    // the declaration / initialisation must be declared in a compiled unit
    // ! This function should not be inlined !
    static Log sInstance{};
    return sInstance;
}

Log::Log()
{
    //TODO: how to check the log has been open
    m_logFile.open("log.txt", std::ios::out);


    std::string projectName = "CudaInterop";

    //get time now
    time_t now = time(NULL);
    tm now_tm = {};
    char str[26] = {};
    localtime_s(&now_tm, &now);
    asctime_s(str, 26, &now_tm);

    m_logFile << "Log file - Project : " << projectName << " - Begin at : " << str << std::endl;

    for (int i = 0; i < 256; i++)
        m_logFile << "-";
    m_logFile << std::endl;
}

Log::~Log()
{
    std::string projectName = "CudaInterop";
   

    for (int i = 0; i < 256; i++)
        m_logFile << "-";
    m_logFile << std::endl;

    //get time now
    time_t now = time(NULL);
    tm now_tm = {};
    char str[26] = {};
    localtime_s(&now_tm, &now);
    asctime_s(str, 26, &now_tm);

    m_logFile << "End Log file - Project : " << projectName << " - End at : " << str << std::endl;
    m_logFile.close();
}

void Log::debugLog(std::string text)
{
    m_logFile << text << std::endl;
}

void Log::debugLogError(std::string text)
{
    //TODO: add line and file
    m_logFile << "Error : " << text << std::endl;
}

