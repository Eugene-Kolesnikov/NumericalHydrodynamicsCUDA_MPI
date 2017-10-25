#ifndef FILELOGGER_HPP
#define FILELOGGER_HPP

#include <fstream>
#include <ctime>
#include <string>

#define _WARNING_ logging::FileLogger::e_logType::LOG_WARNING
#define _ERROR_ logging::FileLogger::e_logType::LOG_ERROR
#define _INFO_ logging::FileLogger::e_logType::LOG_INFO

// Use the namespace you want
namespace logging {

    class FileLogger {
        public:
            enum class e_logType { LOG_ERROR, LOG_WARNING, LOG_INFO };
            explicit FileLogger (int node, std::string fname);
            ~FileLogger ();
            friend FileLogger &operator << (FileLogger &logger, const e_logType l_type);
            friend FileLogger &operator << (FileLogger &logger, const char *text);
            FileLogger (const FileLogger &) = delete;
            FileLogger &operator= (const FileLogger &) = delete;

        private:
            static char* getTime(const struct tm *timeptr);
        private:
            std::ofstream           myFile;
            unsigned int            numWarnings;
            unsigned int            numErrors;
    }; // class end

}  // namespace

std::string createLogFilename(int node);

#endif // FILELOGGER_HPP
