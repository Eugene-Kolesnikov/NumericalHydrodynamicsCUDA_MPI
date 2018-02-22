/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <mpi.h>
#include <ComputationalScheme/include/interface.h>
#include <ComputationalModel/include/ComputationalModel.hpp>

ComputationalModel::ComputationalModel(const char* comp, const char* grid):
    schemeModel(comp), gridModel(grid)
{
    compSchemeLibHandler = nullptr;
    scheme = nullptr;
    field = nullptr;
    tmpCPUField = nullptr;
    lr_halo = nullptr;
    tb_halo = nullptr;
    lrtb_halo = nullptr;
    rcv_lr_halo = nullptr;
    rcv_tb_halo = nullptr;
    rcv_lrtb_halo = nullptr;
    #ifdef __DEBUG__
        dbg_field = nullptr;
        dbg_lr_halo = nullptr;
        dbg_tb_halo = nullptr;
        dbg_lrtb_halo = nullptr;
    #endif
}

ComputationalModel::~ComputationalModel()
{
    if(scheme != nullptr)
        delete scheme;
    libLoader::close(compSchemeLibHandler);
}

void ComputationalModel::createMpiStructType()
{
    const std::type_info& data_typeid = scheme->getDataTypeid();
    int mpi_err_status, resultlen;
    char err_buffer[MPI_MAX_ERROR_STRING];
    MPI_Datatype MPI_DATA_TYPE;
    if(data_typeid == typeid(float)) {
        MPI_DATA_TYPE = MPI_FLOAT;
    } else if(data_typeid == typeid(double)) {
        MPI_DATA_TYPE = MPI_DOUBLE;
    } else if(data_typeid == typeid(long double)) {
        MPI_DATA_TYPE = MPI_LONG_DOUBLE;
    } else {
        throw std::runtime_error("ComputationalModel::createMpiStructType: "
                "Wrong STRUCT_DATA_TYPE!");
    }
    size_t nitems = scheme->getNumberOfElements();
    const size_t* sheme_offsets = scheme->getCellOffsets();
    const size_t* scheme_blocklengths = scheme->getAmountOfArrayMembers();
    int* blocklengths = new int[nitems];
    MPI_Datatype* types = new MPI_Datatype[nitems];
    MPI_Aint* offsets = new MPI_Aint[nitems];
    for(size_t i = 0; i < nitems; ++i) {
        blocklengths[i] = scheme_blocklengths[i];
        types[i] = MPI_DATA_TYPE;
        offsets[i] = sheme_offsets[i];
    }
    mpi_err_status = MPI_Type_create_struct(nitems, blocklengths, offsets,
            types, &MPI_CellType);
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    mpi_err_status = MPI_Type_commit(&MPI_CellType);
    if(mpi_err_status != MPI_SUCCESS) {
        MPI_Error_string(mpi_err_status, err_buffer, &resultlen);
        throw std::runtime_error(err_buffer);
    }
    delete[] blocklengths;
    delete[] types;
    delete[] offsets;
    (*Log) << "MPI structure has been successfully created";
}

void ComputationalModel::initializeField()
{
    if(nodeType != NODE_TYPE::SERVER_NODE)
        throw std::runtime_error("ComputationalModel::initializeField: "
                "This function should not be called by a Computational Node");
    scheme->initField(field, N_X, N_Y, X_MAX, Y_MAX);
}

void* ComputationalModel::getTmpCPUFieldStoragePtr()
{
    return tmpCPUField;
}

void* ComputationalModel::getField()
{
    if(nodeType != NODE_TYPE::SERVER_NODE)
        throw std::runtime_error("ComputationalModel::getField: "
                "This function should not be called by a Computational Node");
    return field;
}

void* ComputationalModel::getCPUHaloPtr(size_t border_type)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("ComputationalModel::getCPUHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datastruct = scheme->getSizeOfDatastruct();
    if(border_type == CU_LEFT_BORDER)
        return lr_halo;
    else if(border_type == CU_RIGHT_BORDER) {
        byte* lr_haloPtr = (byte*)lr_halo;
        byte* r_haloPtr = lr_haloPtr + lN_Y * size_of_datastruct;
        return (void*)r_haloPtr;
    } else if(border_type == CU_TOP_BORDER)
        return tb_halo;
    else if(border_type == CU_BOTTOM_BORDER) {
        byte* tb_haloPtr = (byte*)tb_halo;
        byte* b_haloPtr = tb_haloPtr + lN_X * size_of_datastruct;
        return (void*)b_haloPtr;
    } else
        throw std::runtime_error("ComputationalModel::getCPUHaloPtr: "
                "Wrong border_type");
}

void* ComputationalModel::getCPUDiagHaloPtr(size_t border_type)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("ComputationalModel::getCPUDiagHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datastruct = scheme->getSizeOfDatastruct();
    if(border_type != CU_LEFT_TOP_BORDER && border_type != CU_RIGHT_TOP_BORDER &&
            border_type != CU_LEFT_BOTTOM_BORDER && border_type != CU_RIGHT_BOTTOM_BORDER)
        throw std::runtime_error("ComputationalModel::getCPUDiagHaloPtr: "
                "Wrong border_type");
    byte* lrtb_haloPtr = (byte*)lrtb_halo;
    byte* haloPtr = lrtb_haloPtr + (border_type - CU_LEFT_TOP_BORDER) * size_of_datastruct;
    return (void*)haloPtr;
}

void* ComputationalModel::getTmpCPUHaloPtr(size_t border_type)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("ComputationalModel::getTmpCPUHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datastruct = scheme->getSizeOfDatastruct();
    if(border_type == CU_LEFT_BORDER)
        return rcv_lr_halo;
    else if(border_type == CU_RIGHT_BORDER) {
        byte* rcv_lr_haloPtr = (byte*)rcv_lr_halo;
        byte* rcv_r_haloPtr = rcv_lr_haloPtr + lN_Y * size_of_datastruct;
        return (void*)rcv_r_haloPtr;
    } else if(border_type == CU_TOP_BORDER)
        return rcv_tb_halo;
    else if(border_type == CU_BOTTOM_BORDER) {
        byte* rcv_tb_haloPtr = (byte*)rcv_tb_halo;
        byte* rcv_b_haloPtr = rcv_tb_haloPtr + lN_X * size_of_datastruct;
        return (void*)rcv_b_haloPtr;
    } else
        throw std::runtime_error("ComputationalModel::getTmpCPUHaloPtr: "
                "Wrong border_type");
}

void* ComputationalModel::getTmpCPUDiagHaloPtr(size_t border_type)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("ComputationalModel::getTmpCPUDiagHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datastruct = scheme->getSizeOfDatastruct();
    if(border_type != CU_LEFT_TOP_BORDER && border_type != CU_RIGHT_TOP_BORDER &&
            border_type != CU_LEFT_BOTTOM_BORDER && border_type != CU_RIGHT_BOTTOM_BORDER)
        throw std::runtime_error("ComputationalModel::getTmpCPUDiagHaloPtr: "
                "Wrong border_type");
    byte* rcv_lrtb_haloPtr = (byte*)rcv_lrtb_halo;
    byte* rcv_haloPtr = rcv_lrtb_haloPtr + (border_type - CU_LEFT_TOP_BORDER) * size_of_datastruct;
    return (void*)rcv_haloPtr;
}

void ComputationalModel::setStopMarker()
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("ComputationalModel::setStopMarker: "
                "This function should not be called by the Server Node");
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    byte* markerValue = (byte*)scheme->getMarkerValue();
    byte* tmpCPUFieldPtr = (byte*)tmpCPUField;
    for(size_t s = 0; s < size_of_datatype; ++s) {
        /// Go through all bytes of the STRUCT_DATA_TYPE
        /// Update the byte
        tmpCPUFieldPtr[s] = markerValue[s];
    }
}

bool ComputationalModel::checkStopMarker()
{
    if(nodeType != NODE_TYPE::SERVER_NODE)
        throw std::runtime_error("ComputationalModel::checkStopMarker: "
                "This function should not be called by a Computational Node");
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    byte* markerValue = (byte*)scheme->getMarkerValue();
    byte* fieldPtr = (byte*)field;
    for(size_t s = 0; s < size_of_datatype; ++s) {
        /// Go through all bytes of the STRUCT_DATA_TYPE
        /// Update the byte
        if(fieldPtr[s] != markerValue[s])
            return false;
    }
    return true;
}

void ComputationalModel::setBorders(size_t mpi_node_x, size_t mpi_node_y)
{
    if(mpi_node_x == 0) {
        borders[CU_LEFT_BORDER] = 1;
    } else {
        borders[CU_LEFT_BORDER] = 0;
    }

    if(mpi_node_x == (MPI_NODES_X - 1)) {
        borders[CU_RIGHT_BORDER] = 1;
    } else {
        borders[CU_RIGHT_BORDER] = 0;
    }

    if(mpi_node_y == 0) {
        borders[CU_TOP_BORDER] = 1;
    } else {
        borders[CU_TOP_BORDER] = 0;
    }

    if(mpi_node_y == (MPI_NODES_Y - 1)) {
        borders[CU_BOTTOM_BORDER] = 1;
    } else {
        borders[CU_BOTTOM_BORDER] = 0;
    }
}

void ComputationalModel::initScheme()
{
    /// Create a path to the lib
    std::string libpath = appPath + SystemRegister::CompScheme::name;
    /// Open the library
    compSchemeLibHandler = libLoader::open(libpath);
    (*Log) << "Opened the computational scheme dynamic library";
    /// Load the function
    auto _createScheme = libLoader::resolve<decltype(&createScheme)>(compSchemeLibHandler, SystemRegister::CompScheme::interface);
    /// Initialize the scheme with the loaded function
    scheme = reinterpret_cast<ComputationalScheme*>(_createScheme(schemeModel.c_str(), gridModel.c_str()));
    (*Log) << "Computational scheme has been successfully created";
}

void ComputationalModel::memcpyField(size_t mpi_node_x, size_t mpi_node_y, TypeMemCpy cpyType)
{
    byte* fieldPtr = (byte*)field;
    byte* tmpCPUFieldPtr = (byte*)tmpCPUField;
    size_t size_of_datastruct = scheme->getSizeOfDatastruct();
    size_t global, globalTmp, x0, y0, x1, y1;
    size_t global_shift, global_shiftTmp;
    // calculate the shift for the particular subfield
    x0 = static_cast<size_t>(N_X / MPI_NODES_X * mpi_node_x);
    y0 = static_cast<size_t>(N_Y / MPI_NODES_Y * mpi_node_y);
    for(size_t x = 0; x < lN_X; ++x) {
        /// Go through all x elements of the subfield
        /** Add the shift to the x-component since the subfield can be located
         * somewhere inside the global field */
        x1 = x0 + x;
        for(size_t y = 0; y < lN_Y; ++y) {
            /// Go through all y elements of the subfield
            /** Add the shift to the y-component since the subfield can be located
             * somewhere inside the global field */
            y1 = y0 + y;
            /** Calculate the global shifts for the 'global field' and 'subfield'
             * using the fact that structure consists of size_of_datastruct
             * amount of bytes each. */
            global_shift = (y1 * N_X + x1) * size_of_datastruct;
            global_shiftTmp = (y * lN_X + x) * size_of_datastruct;
            for(size_t s = 0; s < size_of_datastruct; ++s) {
                /// Go through all bytes of the Cell struct
                /** Add shifts for the bytes of the Cell struct */
                global = global_shift + s;
                globalTmp = global_shiftTmp + s;
                /// Update the byte
                if(cpyType == TmpCPUFieldToField)
                    fieldPtr[global] = tmpCPUFieldPtr[globalTmp];
                else // FieldToTmpCPUField
                    tmpCPUFieldPtr[globalTmp] = fieldPtr[global];
            }
        }
    }
}

#ifdef __DEBUG__
void ComputationalModel::cpu_memcpy(void* rcv, void* snd, size_t N)
{
    byte* rcvPtr = (byte*)rcv;
    byte* sndPtr = (byte*)snd;
    size_t size_of_datastruct = scheme->getSizeOfDatastruct();
    size_t shift, global;
    for(size_t i = 0; i < N; ++i) {
        *Log << std::to_string(i);
        /// Go through all elements of the array
        shift = i * size_of_datastruct;
        for(size_t s = 0; s < size_of_datastruct; ++s) {
            /// Go through all bytes of the Cell struct
            /** Add shifts for the bytes of the Cell struct */
            global = shift + s;
            /// Update the byte
            rcvPtr[global] = sndPtr[global];
        }
    }
}

void* ComputationalModel::getDBGField() const
{
    return dbg_field;
}

void* ComputationalModel::getDBGHaloPtr(size_t border_type)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("ComputationalModel::getDBGHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datastruct = scheme->getSizeOfDatastruct();
    if(border_type == CU_LEFT_BORDER)
        return dbg_lr_halo;
    else if(border_type == CU_RIGHT_BORDER) {
        byte* dbg_lr_haloPtr = (byte*)dbg_lr_halo;
        byte* dbg_r_haloPtr = dbg_lr_haloPtr + lN_Y * size_of_datastruct;
        return (void*)dbg_r_haloPtr;
    } else if(border_type == CU_TOP_BORDER)
        return dbg_tb_halo;
    else if(border_type == CU_BOTTOM_BORDER) {
        byte* dbg_tb_haloPtr = (byte*)dbg_tb_halo;
        byte* dbg_b_haloPtr = dbg_tb_haloPtr + lN_X * size_of_datastruct;
        return (void*)dbg_b_haloPtr;
    } else
        throw std::runtime_error("ComputationalModel::getDBGHaloPtr: "
                "Wrong border_type");
}

void* ComputationalModel::getDBGDiagHaloPtr(size_t border_type)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("ComputationalModel::getDBGDiagHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datastruct = scheme->getSizeOfDatastruct();
    if(border_type != CU_LEFT_TOP_BORDER && border_type != CU_RIGHT_TOP_BORDER &&
            border_type != CU_LEFT_BOTTOM_BORDER && border_type != CU_RIGHT_BOTTOM_BORDER)
        throw std::runtime_error("ComputationalModel::getDBGDiagHaloPtr: "
                "Wrong border_type");
    byte* dbg_lrtb_haloPtr = (byte*)dbg_lrtb_halo;
    byte* dbg_haloPtr = dbg_lrtb_haloPtr + (border_type - CU_LEFT_TOP_BORDER) * size_of_datastruct;
    return (void*)dbg_haloPtr;
}
#endif

std::string ComputationalModel::getErrorString()
{
    return errorString;
}

void ComputationalModel::setLog(logging::FileLogger* _Log) {
    Log = _Log;
}

logging::FileLogger* ComputationalModel::getLog() const {
    return Log;
}

void ComputationalModel::setAppPath(std::string app_path) {
    appPath = app_path;
}

std::string ComputationalModel::getAppPath() const {
    return appPath;
}

void ComputationalModel::setNodeType(enum ComputationalModel::NODE_TYPE node_type) {
    nodeType = node_type;
}

ComputationalModel::NODE_TYPE ComputationalModel::getNodeType() const {
    return nodeType;
}

void ComputationalModel::setMPI_NODES_X(size_t val) {
    MPI_NODES_X = val;
}

size_t ComputationalModel::getMPI_NODES_X() const {
    return MPI_NODES_X;
}

void ComputationalModel::setMPI_NODES_Y(size_t val) {
    MPI_NODES_Y = val;
}

size_t ComputationalModel::getMPI_NODES_Y() const {
    return MPI_NODES_Y;
}

void ComputationalModel::setCUDA_X_THREADS(size_t val) {
    CUDA_X_THREADS = val;
}

size_t ComputationalModel::getCUDA_X_THREADS() const {
    return CUDA_X_THREADS;
}

void ComputationalModel::setCUDA_Y_THREADS(size_t val) {
    CUDA_Y_THREADS = val;
}

size_t ComputationalModel::getCUDA_Y_THREADS() const {
    return CUDA_Y_THREADS;
}

void ComputationalModel::setTAU(double val) {
    TAU = val;
}

double ComputationalModel::getTAU() const {
    return TAU;
}

void ComputationalModel::setN_X(size_t val) {
    N_X = val;
}

size_t ComputationalModel::getN_X() const {
    return N_X;
}

void ComputationalModel::setN_Y(size_t val) {
    N_Y = val;
}

size_t ComputationalModel::getN_Y() const {
    return N_Y;
}

void ComputationalModel::setX_MAX(double val)
{
    X_MAX = val;
}

double ComputationalModel::getX_MAX() const
{
    return X_MAX;
}

void ComputationalModel::setY_MAX(double val)
{
    Y_MAX = val;
}

double ComputationalModel::getY_MAX() const
{
    return Y_MAX;
}

void ComputationalModel::setLN_X(size_t val) {
    lN_X = val;
}

size_t ComputationalModel::getLN_X() const {
    return lN_X;
}

void ComputationalModel::setLN_Y(size_t val) {
    lN_Y = val;
}

size_t ComputationalModel::getLN_Y() const {
    return lN_Y;
}

ComputationalScheme* ComputationalModel::getScheme() const
{
    return scheme;
}
