/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <mpi.h>
#include "ComputationalModel.hpp"

ComputationalModel::ComputationalModel(const char* comp, const char* grid):
    schemeModel(comp), gridModel(grid)
{
    compSchemeLibHandle = nullptr;
    createScheme = nullptr;
    scheme = nullptr;
    field = nullptr;
    tmpCPUField = nullptr;
    lr_halo = nullptr;
    tb_halo = nullptr;
    lrtb_halo = nullptr;
    rcv_lr_halo = nullptr;
    rcv_tb_halo = nullptr;
    rcv_lrtb_halo = nullptr;
}

ComputationalModel::~ComputationalModel()
{
    if(scheme != nullptr)
        delete scheme;
    if(compSchemeLibHandle != nullptr)
        dlclose(compSchemeLibHandle);
}

void ComputationalModel::createMpiStructType()
{
    const std::type_info& data_typeid = scheme->getDataTypeid();
    size_t size_of_datatype = scheme->getSizeOfDatatype();
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
    int* blocklengths = new int[nitems];
    MPI_Datatype* types = new MPI_Datatype[nitems];
    MPI_Aint* offsets = new MPI_Aint[nitems];
    for(size_t i = 0; i < nitems; ++i) {
        blocklengths[i] = 1;
        types[i] = MPI_DATA_TYPE;
        offsets[i] = i * size_of_datatype;
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
    scheme->initField(field, N_X, N_Y);
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
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    if(border_type == LEFT_BORDER)
        return lr_halo;
    else if(border_type == RIGHT_BORDER) {
        byte* lr_haloPtr = (byte*)lr_halo;
        byte* r_haloPtr = lr_haloPtr + lN_Y * nitems * size_of_datatype;
        return (void*)r_haloPtr;
    } else if(border_type == TOP_BORDER)
        return tb_halo;
    else if(border_type == BOTTOM_BORDER) {
        byte* tb_haloPtr = (byte*)tb_halo;
        byte* b_haloPtr = tb_haloPtr + lN_X * nitems * size_of_datatype;
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
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    if(border_type != LEFT_TOP_BORDER && border_type != RIGHT_TOP_BORDER &&
            border_type != LEFT_BOTTOM_BORDER && border_type != RIGHT_BOTTOM_BORDER)
        throw std::runtime_error("ComputationalModel::getCPUDiagHaloPtr: "
                "Wrong border_type");
    byte* lrtb_haloPtr = (byte*)lrtb_halo;
    byte* haloPtr = lrtb_haloPtr + border_type * nitems * size_of_datatype;
    return (void*)haloPtr;
}

void* ComputationalModel::getTmpCPUHaloPtr(size_t border_type)
{
    if(nodeType != NODE_TYPE::COMPUTATIONAL_NODE)
        throw std::runtime_error("ComputationalModel::getTmpCPUHaloPtr: "
                "This function should not be called by the Server Node");
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    if(border_type == LEFT_BORDER)
        return rcv_lr_halo;
    else if(border_type == RIGHT_BORDER) {
        byte* rcv_lr_haloPtr = (byte*)rcv_lr_halo;
        byte* rcv_r_haloPtr = rcv_lr_haloPtr + lN_Y * nitems * size_of_datatype;
        return (void*)rcv_r_haloPtr;
    } else if(border_type == TOP_BORDER)
        return rcv_tb_halo;
    else if(border_type == BOTTOM_BORDER) {
        byte* rcv_tb_haloPtr = (byte*)rcv_tb_halo;
        byte* rcv_b_haloPtr = rcv_tb_haloPtr + lN_X * nitems * size_of_datatype;
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
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    if(border_type != LEFT_TOP_BORDER && border_type != RIGHT_TOP_BORDER &&
            border_type != LEFT_BOTTOM_BORDER && border_type != RIGHT_BOTTOM_BORDER)
        throw std::runtime_error("ComputationalModel::getTmpCPUDiagHaloPtr: "
                "Wrong border_type");
    byte* rcv_lrtb_haloPtr = (byte*)rcv_lrtb_halo;
    byte* rcv_haloPtr = rcv_lrtb_haloPtr + border_type * nitems * size_of_datatype;
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
        borders[LEFT_BORDER] = 1;
    } else {
        borders[LEFT_BORDER] = 0;
    }

    if(mpi_node_x == (MPI_NODES_X - 1)) {
        borders[RIGHT_BORDER] = 1;
    } else {
        borders[RIGHT_BORDER] = 0;
    }

    if(mpi_node_y == 0) {
        borders[TOP_BORDER] = 1;
    } else {
        borders[TOP_BORDER] = 0;
    }

    if(mpi_node_y == (MPI_NODES_Y - 1)) {
        borders[BOTTOM_BORDER] = 1;
    } else {
        borders[BOTTOM_BORDER] = 0;
    }
}

void ComputationalModel::initScheme()
{
    /// Create a path to the lib
    std::string libpath = appPath + "libComputationalScheme.1.0.0.dylib";
    /// Open the library
    compSchemeLibHandle = dlopen(libpath.c_str(), RTLD_LOCAL | RTLD_LAZY);
    if (compSchemeLibHandle == nullptr)
        throw std::runtime_error(dlerror());
    else
        (*Log) << "Opened the computational scheme dynamic library";
    /// Load the function
    createScheme = (void* (*)(const char*,const char*))dlsym(compSchemeLibHandle, "createScheme");
    if(createScheme == nullptr) {
        throw std::runtime_error("Can't load the function from the Computational scheme library!");
    }

    /// Initialize the scheme with the loaded function
    scheme = (ComputationalScheme*)createScheme(schemeModel.c_str(), gridModel.c_str());
    (*Log) << "Computational scheme has been successfully created";
}

void ComputationalModel::memcpyField(size_t mpi_node_x, size_t mpi_node_y, TypeMemCpy cpyType)
{
    byte* fieldPtr = (byte*)field;
    byte* tmpCPUFieldPtr = (byte*)tmpCPUField;
    size_t size_of_datatype = scheme->getSizeOfDatatype();
    size_t nitems = scheme->getNumberOfElements();
    size_t global, globalTmp, x0, y0, x1, y1;
    size_t global_shift, global_shiftTmp;
    size_t global_shift_item, global_shiftTmp_item;
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
             * using the fact that structure consists of nitems amount of
             * elements which are size_of_datatype amount of bytes each. */
            global_shift = (y1 * N_X + x1) * nitems * size_of_datatype;
            global_shiftTmp = (y * lN_X + x) * nitems * size_of_datatype;
            for(size_t i = 0; i < nitems; ++i) {
                /// Go through all elements of the Cell
                /** Add shifts for the elements inside the Cell structure */
                global_shift_item = global_shift + i * size_of_datatype;
                global_shiftTmp_item = global_shiftTmp + i * size_of_datatype;
                for(size_t s = 0; s < size_of_datatype; ++s) {
                    /// Go through all bytes of the STRUCT_DATA_TYPE
                    /** Add shifts for the bytes of the STRUCT_DATA_TYPE */
                    global = global_shift_item + s;
                    globalTmp = global_shiftTmp_item + s;
                    /// Update the byte
                    if(cpyType == TmpCPUFieldToField)
                        fieldPtr[global] = tmpCPUFieldPtr[globalTmp];
                    else // FieldToTmpCPUField
                        tmpCPUFieldPtr[globalTmp] = fieldPtr[global];
                }
            }
        }
    }
}

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
