#include <iostream>
#include "FileLogger.hpp"
#include <exception>

namespace logging {

FileLogger::FileLogger(size_t globalMPI_id)
      :numWarnings(0), numErrors(0), numDebugMsgs(0), mpi_id(globalMPI_id)
{
}

FileLogger::~FileLogger () {
    if (myFile.is_open()) {
        myFile << "Errors: " << numErrors
            << "\nWarnings: " << numWarnings
            << "\nDebug msgs: " << numDebugMsgs << std::endl;
        myFile.close();
    }
}

FileLogger &operator << (FileLogger &logger, const FileLogger::e_logType l_type) {
    switch (l_type) {
        case logging::FileLogger::e_logType::LOG_ERROR:
            logger.myFile << "[ERROR]: ";
            ++logger.numErrors;
            break;
        case logging::FileLogger::e_logType::LOG_WARNING:
            logger.myFile << "[WARNING]: ";
            ++logger.numWarnings;
            break;
        case logging::FileLogger::e_logType::LOG_DEBUG:
            logger.myFile << "[DEBUG]: ";
            ++logger.numDebugMsgs;
            break;
        default:
            logger.myFile << "[INFO]: ";
            break;
    } // sw
    return logger;
}

FileLogger &operator << (FileLogger &logger, const char *text) {
    std::time_t result = std::time(nullptr);
    logger.myFile << "("
                  << FileLogger::getTime(std::localtime(&result))
                  << ") " << text << std::endl;
    return logger;
}

FileLogger &operator << (FileLogger &logger, std::string text)
{
    std::time_t result = std::time(nullptr);
    logger.myFile << "("
                  << FileLogger::getTime(std::localtime(&result))
                  << ") " << text << std::endl;
    return logger;
}

char* FileLogger::getTime(const struct tm *timeptr)
{
  static char result[10];
  sprintf(result, "%.2d:%.2d:%.2d",
    timeptr->tm_hour, timeptr->tm_min, timeptr->tm_sec);
  return result;
}

void FileLogger::openLogFile(std::string appPath)
{
    myFile.open(createLogFilename(appPath));
    std::time_t result = std::time(nullptr);
    // Write the first lines
    if (myFile.is_open()) {
        myFile << "Log file created: "
               << getTime(std::localtime(&result))
               << std::endl;
    } else {
        std::string error_msg = "Node (" + std::to_string(mpi_id) +
                "): Can't create a log file!";
        throw std::runtime_error(error_msg);
    }
}

std::string FileLogger::createLogFilename(std::string appPath)
{
    std::string filename = appPath + "log/node_" + std::to_string(mpi_id) + ".log";
    return filename;
}

bool FileLogger::is_open()
{
    return myFile.is_open();
}

}
