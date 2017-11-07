#ifndef FILELOGGER_HPP
#define FILELOGGER_HPP

#include <fstream>
#include <ctime>
#include <string>
#include <cstdlib>

#define _WARNING_ logging::FileLogger::e_logType::LOG_WARNING
#define _ERROR_ logging::FileLogger::e_logType::LOG_ERROR
#define _INFO_ logging::FileLogger::e_logType::LOG_INFO

namespace logging {

    class FileLogger {
        public:
            enum class e_logType { LOG_ERROR, LOG_WARNING, LOG_INFO };
            explicit FileLogger (size_t globalMPI_id);
            ~FileLogger ();
            friend FileLogger &operator << (FileLogger &logger, const e_logType l_type);
            friend FileLogger &operator << (FileLogger &logger, const char *text);
            friend FileLogger &operator << (FileLogger &logger, std::string text);
            FileLogger (const FileLogger &) = delete;
            FileLogger &operator= (const FileLogger &) = delete;
            
            void openLogFile();
            std::string createLogFilename();
            bool is_open();

        private:
            static char* getTime(const struct tm *timeptr);
            
        private:
            std::ofstream myFile;
            size_t numWarnings;
            size_t numErrors;
            size_t mpi_id;
    }; // class end

}

#endif // FILELOGGER_HPP