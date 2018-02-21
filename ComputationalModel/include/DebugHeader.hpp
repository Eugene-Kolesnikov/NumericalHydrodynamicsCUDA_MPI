#ifndef DEBUGHEADER_HPP
#define DEBUGHEADER_HPP

/**
 * __DEBUG__ is a marker which indicates that the debugging must be performed.
 * If the debugging must be excluded from the program, the line
 * '#define __DEBUG__' must be commented.
 */
//#define __DEBUG__

/**
 * __DEBUG__UPLOAD_DBG_ARRAYS is a marker which indicates that the debugging arrays
 * must be uploaded instead of the GPU computed once. If the uploading must be excluded
 * from the program, the line '#define __DEBUG__UPLOAD_DBG_ARRAYS' must be commented.
 */
#define __DEBUG__UPLOAD_DBG_ARRAYS

/**
 * __DEBUG__RELOAD_CORRECT is a marker which tells if the correct value from
 * the debugger function (which was computed on the CPU side) must be uploaded
 * to the GPU instead of the incorrect one on the GPU.
 * '1' stands for 'reload' and '0' -- for 'do not reload'.
 */
#define __DEBUG__RELOAD_CORRECT 0

/**
 * __DEBUG_CHECK_ARRAYS is a marker which indicates wheather computed GPU arrays
 * and dbg arrays must be compared with each other or not.
 * '1' stands for 'must be compared' and '0' -- for 'must not be compared'.
 */
#define __DEBUG_CHECK_ARRAYS 0

/**
 * __DEBUG_PRINT_ARRAYS is a marker which indicates wheather computed GPU arrays
 * and dbg arrays must be printed in a log file or not.
 * '1' stands for 'must be printed' and '0' -- for 'must not be printed'.
 */
#define __DEBUG_PRINT_ARRAYS 0


#ifdef __DEBUG__
    #define _DEBUG_SUCCESS_ 1
    #define _DEBUG_FAILURE_ 0
    #define BYTE_TYPE char

    #define _DEBUG_PRINT_MSG_(func, msg, Log) do {                              \
        Log << _LOG_DEBUG_ << (std::string("(") + std::string(func) +           \
            std::string(") in file '") + std::string(__FILE__) +                \
            std::string("' in line ") + std::to_string(__LINE__) +              \
            std::string(": ") + std::string(msg) + std::string("!"));           \
    } while(0)

    #define _DEBUG_PRINT_ARRAYS_CPU_GPU(cpu, gpu, N, SS, Log) do {              \
        if(__DEBUG_PRINT_ARRAYS == 0) break;                                    \
        HANDLE_CUERROR(cudaDeviceSynchronize());                                \
        BYTE_TYPE* p1 = (BYTE_TYPE*)cpu;                                        \
        BYTE_TYPE* p2 = new BYTE_TYPE[N * SS];                                  \
        double* p = nullptr;                                                    \
        cudaError err = cudaMemcpy(p2, gpu, N * SS, cudaMemcpyDeviceToHost);    \
        if(err != cudaSuccess) {                                                \
            _DEBUG_PRINT_MSG_("_DEBUG_PRINT_ARRAYS_CPU_GPU",                    \
                cudaGetErrorString(err), Log);                                  \
            delete[] p2; throw 1;                                                            \
            break;                                                 \
        }                                                                       \
        std::string arr1, arr2;                                                 \
        for(size_t i = 0; i < N; ++i) {                                         \
            p = (double*)(p1 + i * SS);                                         \
            arr1 += (std::to_string(*p) + std::string(" "));                    \
            p = (double*)(p2 + i * SS);                                         \
            arr2 += (std::to_string(*p) + std::string(" "));                    \
        }                                                                       \
        Log << _LOG_DEBUG_ << arr1;                                             \
        Log << _LOG_DEBUG_ << arr2;                                             \
        delete[] p2;                                                            \
    } while(0)

    #define _DEBUG_PRINT_ARRAYS_CPU_CPU(cpu1, cpu2, N, SS, Log) do {            \
        if(__DEBUG_PRINT_ARRAYS == 0) break;                                    \
        HANDLE_CUERROR(cudaDeviceSynchronize());                                \
        BYTE_TYPE* p1 = (BYTE_TYPE*)cpu1;                                       \
        BYTE_TYPE* p2 = (BYTE_TYPE*)cpu2;                                       \
        double* p = nullptr;                                                    \
        double t;                                                               \
        std::string arr1, arr2, dist;                                           \
        for(size_t i = 0; i < N; ++i) {                                         \
            p = (double*)(p1 + i * SS);                                         \
            arr1 += (std::to_string(*p) + std::string(" "));                    \
            t = *p;                                                             \
            p = (double*)(p2 + i * SS);                                         \
            arr2 += (std::to_string(*p) + std::string(" "));                    \
            dist += (std::to_string(std::abs(t - *p)) + std::string(" "));      \
        }                                                                       \
        Log << _LOG_DEBUG_ << arr1;                                             \
        Log << _LOG_DEBUG_ << arr2;                                             \
        Log << _LOG_DEBUG_ << dist;                                             \
    } while(0)

    #define _DEBUG_PRINT_FIELDS_CPU_GPU(cpu, gpu, NX, NY, SS, Log) do {         \
        if(__DEBUG_PRINT_ARRAYS == 0) break;                                    \
        HANDLE_CUERROR(cudaDeviceSynchronize());                                \
        BYTE_TYPE* p1 = (BYTE_TYPE*)cpu;                                        \
        BYTE_TYPE* p2 = new BYTE_TYPE[NX*NY*SS];                                \
        BYTE_TYPE* gpu1 = (BYTE_TYPE*)gpu;                                      \
        double* p = nullptr;                                                    \
        cudaError err = cudaMemcpy(p2, gpu1, NX*NY*SS, cudaMemcpyDeviceToHost); \
        if(err != cudaSuccess) {                                                \
            _DEBUG_PRINT_MSG_("CHECK_CPU_GPU_ARRAYS_EQUALITY",                  \
                cudaGetErrorString(err), Log);                                  \
            delete[] p2; throw 1;                                                            \
            break;                                                 \
        }                                                                       \
        std::string arr1, arr2;                                                 \
        for(size_t y = 0; y < NY; ++y) {                                        \
            arr1 = ""; arr2 = "";                                               \
            for(size_t x = 0; x < NX; ++x) {                                    \
                p = (double*)(p1 + (y * NX + x) * SS);                          \
                arr1 += (std::to_string(*p) + std::string(" "));                \
                p = (double*)(p2 + (y * NX + x) * SS);                          \
                arr2 += (std::to_string(*p) + std::string(" "));                \
            }                                                                   \
            arr1 += "   |   ";                                                  \
            arr1 += arr2;                                                       \
            Log << _LOG_DEBUG_ << arr1;                                         \
        }                                                                       \
        delete[] p2;                                                            \
    } while(0)

    #define CHECK_CPU_CPU_ARRAYS_EQUALITY_BYTES(ptr1, ptr2, size, Log) do {     \
        if(__DEBUG_CHECK_ARRAYS == 0) break;                                    \
        BYTE_TYPE* p1 = (BYTE_TYPE*)ptr1;                                       \
        BYTE_TYPE* p2 = (BYTE_TYPE*)ptr2;                                       \
        bool debug_result = _DEBUG_SUCCESS_;                                    \
        for(size_t i = 0; i < size; ++i) {                                      \
            if(p1[i] != p2[i]) {                                                \
                debug_result = _DEBUG_FAILURE_;                                 \
            }                                                                   \
        }                                                                       \
        if(debug_result == _DEBUG_FAILURE_) {                                   \
            _DEBUG_PRINT_MSG_("CHECK_CPU_CPU_ARRAYS_EQUALITY",                  \
                "Arrays are not equal", Log);                                   \
            throw 1;                                                            \
        }                                                                       \
    } while(0)

    #define CHECK_CPU_CPU_ARRAYS_EQUALITY_DOUBLES(ptr1, ptr2, N, SS, Log) do {  \
        if(__DEBUG_CHECK_ARRAYS == 0) break;                                    \
        BYTE_TYPE* p1 = (BYTE_TYPE*)ptr1;                                       \
        BYTE_TYPE* p2 = (BYTE_TYPE*)ptr2;                                       \
        double* p1p = nullptr;                                                  \
        double* p2p = nullptr;                                                  \
        bool debug_result = _DEBUG_SUCCESS_;                                    \
        /*std::string arr1;*/                                                   \
        for(size_t i = 0; i < N; ++i) {                                         \
            p1p = (double*)(p1 + i * SS);                                       \
            p2p = (double*)(p2 + i * SS);                                       \
            /*arr1 += std::to_string(std::abs(*p1p - *p2p));*/                  \
            /*arr1 += " ";*/                                                    \
            if(std::abs(*p1p - *p2p) > 1e-6) {                                  \
                debug_result = _DEBUG_FAILURE_;                                 \
            }                                                                   \
        }                                                                       \
        /*Log << _LOG_DEBUG_ << arr1;*/                                         \
        if(debug_result == _DEBUG_FAILURE_) {                                   \
            _DEBUG_PRINT_MSG_("CHECK_CPU_CPU_ARRAYS_EQUALITY",                  \
                "Arrays are not equal", Log);                                   \
            throw 1;                                                            \
        }                                                                       \
    } while(0)

    #define CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(cpu, gpu, shift, size, Log) do {\
        if(__DEBUG_CHECK_ARRAYS == 0) break;                                    \
        HANDLE_CUERROR(cudaDeviceSynchronize());                                \
        BYTE_TYPE* p1 = ((BYTE_TYPE*)cpu) + shift;                              \
        BYTE_TYPE* p2 = new BYTE_TYPE[size];                                    \
        BYTE_TYPE* gpu1 = ((BYTE_TYPE*)gpu) + shift;                            \
        cudaError err = cudaMemcpy(p2, gpu1, size, cudaMemcpyDeviceToHost);     \
        if(err != cudaSuccess) {                                                \
            _DEBUG_PRINT_MSG_("CHECK_CPU_GPU_ARRAYS_EQUALITY",                  \
                cudaGetErrorString(err), Log);                                  \
            delete[] p2; throw 1;                                                            \
            break;                                                 \
        }                                                                       \
        bool debug_result = _DEBUG_SUCCESS_;                                    \
        for(size_t i = 0; i < size; ++i) {                                      \
            if(p1[i] != p2[i]) {                                                \
                debug_result = _DEBUG_FAILURE_;                                 \
            }                                                                   \
        }                                                                       \
        if(debug_result == _DEBUG_FAILURE_) {                                   \
            _DEBUG_PRINT_MSG_("CHECK_CPU_GPU_ARRAYS_EQUALITY",                  \
                "Arrays are not equal", Log);                                   \
            throw 1;                                                            \
            if(__DEBUG__RELOAD_CORRECT == 1) {                                  \
                err = cudaMemcpy(gpu, cpu, size, cudaMemcpyHostToDevice);       \
                if(err != cudaSuccess) {                                        \
                    _DEBUG_PRINT_MSG_("CHECK_CPU_GPU_ARRAYS_EQUALITY",          \
                        cudaGetErrorString(err), Log);                          \
                    delete[] p2; break;                                         \
                }                                                               \
                _DEBUG_PRINT_MSG_("CHECK_CPU_CPU_ARRAYS_EQUALITY",              \
                    "Debbuged array is uploaded", Log);                         \
            }                                                                   \
        }                                                                       \
        delete[] p2;                                                            \
    } while(0)

    #define CHECK_GPU_CPU_ARRAYS_EQUALITY_BYTES(gpu, cpu, shift, size, Log)     \
        CHECK_CPU_GPU_ARRAYS_EQUALITY_BYTES(cpu, gpu, shift, size, Log)

    #define CHECK_GPU_GPU_ARRAYS_EQUALITY_BYTES(gpu1, gpu2, size, Log) do {     \
        if(__DEBUG_CHECK_ARRAYS == 0) break;                                    \
        HANDLE_CUERROR(cudaDeviceSynchronize());                                \
        BYTE_TYPE* p1 = new BYTE_TYPE[size];                                    \
        BYTE_TYPE* p2 = new BYTE_TYPE[size];                                    \
        cudaError err = cudaMemcpy(p1, gpu1, size, cudaMemcpyDeviceToHost);     \
        if(err != cudaSuccess) {                                                \
            _DEBUG_PRINT_MSG_("CHECK_GPU_GPU_ARRAYS_EQUALITY",                  \
                cudaGetErrorString(err), Log);                                  \
            delete[] p1; delete[] p2; break;                                    \
        }                                                                       \
        err = cudaMemcpy(p2, gpu2, size, cudaMemcpyDeviceToHost);               \
        if(err != cudaSuccess) {                                                \
            _DEBUG_PRINT_MSG_("CHECK_GPU_GPU_ARRAYS_EQUALITY",                  \
                cudaGetErrorString(err), Log);                                  \
            delete[] p1; delete[] p2; break;                                    \
        }                                                                       \
        bool debug_result = _DEBUG_SUCCESS_;                                    \
        for(size_t i = 0; i < size; ++i) {                                      \
            if(p1[i] != p2[i]) {                                                \
                debug_result = _DEBUG_FAILURE_;                                 \
            }                                                                   \
        }                                                                       \
        if(debug_result == _DEBUG_FAILURE_) {                                   \
            _DEBUG_PRINT_MSG_("CHECK_CPU_GPU_ARRAYS_EQUALITY",                  \
                "Arrays are not equal", Log);                                   \
            throw 1;                                                            \
        }                                                                       \
        delete[] p1; delete[] p2;                                               \
    } while(0)
#endif

#endif // DEBUGHEADER_HPP
