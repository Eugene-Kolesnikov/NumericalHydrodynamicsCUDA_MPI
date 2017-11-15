#ifndef GPU_STATUS_H
#define GPU_STATUS_H

typedef int ErrorStatus;
#define GPU_SUCCESS (0)
#define GPU_ERROR (-1)

#define HANDLE_GPUERROR(call) {                            \
    ErrorStatus err = call;                                \
    if(err != GPU_SUCCESS) {                               \
        std::string error = std::string("GPU error: ") +   \
            model->getErrorString();                       \
        throw std::runtime_error(error);                   \
    }                                                      \
} while (0)

#define CM_HANDLE_GPUERROR(call) {                         \
    ErrorStatus err = call;                                \
    if(err != GPU_SUCCESS) {                               \
        errorString = scheme->getErrorString();            \
        return GPU_ERROR;                                  \
    }                                                      \
} while (0)

#define CM_HANDLE_GPUERROR_PTR(ptr) {                      \
    if(ptr == nullptr) {                                   \
        errorString = scheme->getErrorString();            \
        return GPU_ERROR;                                  \
    }                                                      \
} while (0)

#endif // GPU_STATUS_H
