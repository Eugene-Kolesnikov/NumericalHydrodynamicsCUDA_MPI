/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ComputationalModel.cpp
 * Author: eugene
 * 
 * Created on November 1, 2017, 12:51 PM
 */

#include "ComputationalModel.hpp"

MPI_Datatype ComputationalModel::MPI_CellType;

ComputationalModel::ComputationalModel() 
{
}

ComputationalModel::~ComputationalModel() 
{
}

void ComputationalModel::setAppPath(std::string app_path)
{
    appPath = app_path;
}

std::string ComputationalModel::getAppPath() const
{
    return appPath;
}

void ComputationalModel::setNodeType(enum NODE_TYPE node_type) {
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