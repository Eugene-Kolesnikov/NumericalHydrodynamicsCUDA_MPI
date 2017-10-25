#include <iostream>
#include "FileLogger.hpp"

namespace logging {

FileLogger::FileLogger(int node, std::string fname)
      :numWarnings (0U), numErrors (0U)
{
    myFile.open (fname);
    std::time_t result = std::time(nullptr);
    // Write the first lines
    if (myFile.is_open()) {
        myFile << "Log file created: "
               << getTime(std::localtime(&result))
               << std::endl;
    } else {
        std::cout << "ERROR: Node " << node << " can't open log file!";
        exit(2);
    }
}

FileLogger::~FileLogger () {
    if (myFile.is_open()) {
        myFile.close();
    } // if
}

FileLogger &operator << (FileLogger &logger, const FileLogger::e_logType l_type) {
    std::time_t result = std::time(nullptr);
    switch (l_type) {
        case logging::FileLogger::e_logType::LOG_ERROR:
            logger.myFile << "[ERROR]: ";
            ++logger.numErrors;
            break;
        case logging::FileLogger::e_logType::LOG_WARNING:
            logger.myFile << "[WARNING]: ";
            ++logger.numWarnings;
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

char* FileLogger::getTime(const struct tm *timeptr)
{
  static char result[10];
  sprintf(result, "%.2d:%.2d:%.2d",
    timeptr->tm_hour, timeptr->tm_min, timeptr->tm_sec);
  return result;
}

}

std::string createLogFilename(int node)
{
    std::string filename = "log/node" + std::to_string(node) + ".log";
    return filename;
}
