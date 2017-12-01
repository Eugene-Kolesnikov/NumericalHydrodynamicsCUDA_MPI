#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "Datareader.hpp"
#include "ui_datareader.h"
#include <QString>
#include <fstream>
#include <cstdlib>
#include <vector>

class Visualizer
{
    typedef char byte;
public:
    Visualizer(Ui::DataReader* _ui, QCPColorMap* _colorMap, QCPColorScale* _colorScale);
    ~Visualizer();

    bool readReportFile(QString path);
    void renderFrame(size_t t);
    QSize getWindowSize();
    size_t getTotalSteps() const;
    void removeComboBoxItems();

protected:
    void setProgress(double val);
    bool readEnvironment();
    void updateColorMap(size_t t);

private:
    Ui::DataReader* ui;
    QCPColorMap* colorMap;
    QCPColorScale* colorScale;
    std::fstream file;

protected:
    size_t MPI_NODES_X;
    size_t MPI_NODES_Y;
    size_t CUDA_X_THREADS;
    size_t CUDA_Y_THREADS;
    double TAU;
    double TOTAL_TIME;
    double STEP_LENGTH;
    size_t N_X;
    size_t N_Y;
    size_t X_MAX;
    size_t Y_MAX;

protected:
    size_t size_of_datatype;
    size_t nitems;
    size_t params;

protected:
    // timestep-property-field
    std::vector<std::vector<std::vector<double>>> Field;
    size_t curProperty;
    size_t totalSteps;

protected:
    bool initialized;
};

#endif // VISUALIZER_HPP
