#include "Simplevisualizer.hpp"
#include <QDateTime>
#include <exception>
#include <typeinfo>

SimpleVisualizer::SimpleVisualizer(logging::FileLogger* _Log):
    Visualizer(_Log)
{
    colorMap = nullptr;
    colorScale = nullptr;
    marginGroup = nullptr;
    app = static_cast<QApplication*>(QApplication::instance());
    *Log << "Constructed SimpleVisualizer";
}

SimpleVisualizer::~SimpleVisualizer()
{
}

void SimpleVisualizer::initVisualizer()
{
    *Log << "Initialization of the visualizer";
    openReportFile();
    initQtEnv();
    initCustomPlot();
    window.show();
    *Log << "Opened window";
    app->processEvents();
    *Log << "QApplication has processed pending events";
}

void SimpleVisualizer::openReportFile()
{
    std::string time = QDateTime::currentDateTime().toString().toStdString();
    report = appPath + "report." + time + ".data";
    file.open(report.c_str(), std::fstream::out);
    if(file.is_open() == false)
        throw std::runtime_error("Can't create the report file!");
    *Log << "Opened report file";
    writeEnvironment();
}

void SimpleVisualizer::initQtEnv()
{
    *Log << "Initialization of the Qt environment";
    window.setWindowTitle("Simulation computation");
    window.setProgress(0);
    window.setEnvironment(MPI_NODES_X, MPI_NODES_Y, CUDA_X_THREADS,
        CUDA_Y_THREADS, TAU, TOTAL_TIME, STEP_LENGTH, N_X, N_Y, X_MAX, Y_MAX);
    window.setWindowSize(X_MAX, Y_MAX);
    *Log << "Initialized Qt environment";
}

void SimpleVisualizer::initCustomPlot()
{
    *Log << "Initialization of the QCustomPlot";
    customPlot = window.getCustomPlot();
    // configure axis rect:
    customPlot->axisRect()->setupFullAxesBox(true);
    customPlot->xAxis->setLabel("x");
    customPlot->yAxis->setLabel("y");
    // set up the QCPColorMap:
    colorMap = new QCPColorMap(customPlot->xAxis, customPlot->yAxis);
    colorMap->data()->setSize(N_X, N_Y);
    // Span the coordinate range appropriately
    colorMap->data()->setRange(QCPRange(0, X_MAX), QCPRange(0, Y_MAX));
    colorScale = new QCPColorScale(customPlot);
    // add it to the right of the main axis rect
    customPlot->plotLayout()->addElement(0, 1, colorScale);
    // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorScale->setType(QCPAxis::atRight);
    // associate the color map with the color scale
    colorMap->setColorScale(colorScale);
    colorScale->axis()->setLabel(params->begin()->propertyName.c_str());
    // set the color gradient of the color map to one of the presets:
    colorMap->setGradient(QCPColorGradient::gpJet);
    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    marginGroup = new QCPMarginGroup(customPlot);
    customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
    colorMap->rescaleDataRange();
    // rescale the key (x) and value (y) axes so the whole color map is visible:
    customPlot->rescaleAxes();
    *Log << "Initialized QCustomPlot";
}

void SimpleVisualizer::renderFrame(void* field)
{
    std::type_index data_type_index = (*params)[0].typeInfo;
    if(data_type_index == std::type_index(typeid(float))) {
        writeFrame<float>(field);
        updateColorMap<float>(field);
    } else if(data_type_index == std::type_index(typeid(double))) {
        writeFrame<double>(field);
        updateColorMap<double>(field);
    } else if(data_type_index == std::type_index(typeid(long double))) {
        writeFrame<long double>(field);
        updateColorMap<long double>(field);
    } else {
        throw std::runtime_error("SimpleVisualizer::renderFrame: Unknown data type!");
    }
    *Log << "rendered frame";
}

void SimpleVisualizer::setProgress(double val)
{
    window.setProgress(val);
    *Log << "set progress value";
}

void SimpleVisualizer::deinitVisualizer()
{
    file.close();
}

void SimpleVisualizer::writeEnvironment()
{
    file << MPI_NODES_X << ' ' << MPI_NODES_Y << ' ' << CUDA_X_THREADS << ' ' << CUDA_Y_THREADS
         << ' ' << TAU << ' ' << TOTAL_TIME << ' ' << STEP_LENGTH << ' ' << N_X << ' ' << N_Y
         << ' ' << X_MAX << ' ' << Y_MAX << ' ' << params->size();
    for(auto it = params->begin(); it != params->end(); ++it) {
        file << ' ' << it->propertyName << ' ' << it->variables;
    }
    file << ' ';
    *Log << "saved environment to the report file";
}

template<typename T> void SimpleVisualizer::writeFrame(void* field)
{
    byte* cfield = (byte*)field; // pointer to the first Cell
    byte* tcfield = nullptr; // current Cell address
    size_t offset; // offset of a particular element inside the Cell
    size_t items = params->size(); // total amount of parameters
    size_t fieldSize = N_X * N_Y;
    size_t totalVars; // total amount of variables of an element
    for(size_t k = 0; k < items; ++k) { // go through all elements
        offset = (*params)[k].offset; // save the offset
        totalVars = (*params)[k].variables; // save the amount of variables of the element
        if(totalVars != 1) {
            throw std::runtime_error("SimpleVisualizer::writeFrame: Unsupported "
                                     "functionality (too many variables in an element)!");
        }
        // write the field of this elements
        for(size_t i = 0; i < fieldSize; ++i) { // go through all Cells of the field
            tcfield = cfield + i * size_of_datastruct; // get the address of the current Cell
            // go through all variable of the element
            for(size_t var = 0; var < totalVars; ++var) {
                // write down the variable
                file << *((T*)(tcfield + offset) + var) << ' ';
            }
        }
    }
}

template<typename T> void SimpleVisualizer::updateColorMap(void* field)
{
    byte* cfield = (byte*)field; // pointer to the first Cell
    byte* tcfield = nullptr; // current Cell address
    size_t offset = (*params)[0].offset;
    size_t totalVars = (*params)[0].variables;
    for(size_t x = 0; x < N_X; ++x) {
        for(size_t y = 0; y < N_Y; ++y) {
            if(totalVars == 1) {
                tcfield = cfield + (y * N_X + x) * size_of_datastruct;
                colorMap->data()->setCell(x, y, *((T*)(tcfield + offset)));
            } else {
                throw std::runtime_error("SimpleVisualizer::updateColorMap: Unsupported "
                                         "functionality (too many variables in an element)!");
            }
        }
    }
    *Log << "Updated the ColorMap";
    colorMap->rescaleDataRange();
    *Log << "Rescaled data range";
    customPlot->replot();
    *Log << "CustomPlot replotted";
    app->processEvents();
    *Log << "QApplication has processed pending events";
}
