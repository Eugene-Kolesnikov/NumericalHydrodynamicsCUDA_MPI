#ifndef SIMPLEVISUALIZER_H
#define SIMPLEVISUALIZER_H

#include "MainWindow.hpp"
#include <QApplication>
#include "Visualizer.hpp"
#include "Qcustomplot.hpp"
#include <fstream>
#include <thread>
#include <iostream>

class SimpleVisualizer : public Visualizer
{
    typedef char byte;
public:
    SimpleVisualizer(logging::FileLogger* _Log);
    virtual ~SimpleVisualizer();

public:
    virtual void initVisualizer() override;
    virtual void renderFrame(void* field) override;
    virtual void setProgress(double val) override;
    virtual void deinitVisualizer() override;

protected:
    void openReportFile();
    void initQtEnv();
    void initCustomPlot();

protected:
    void writeEnvironment();
    template<typename T> void writeFrame(void* field);
    template<typename T> void updateColorMap(void* field);

protected:
    string report;

public:
    MainWindow window;
    QApplication* app;

protected:
    fstream file;
    QCustomPlot* customPlot;
    QCPColorMap* colorMap;
    QCPColorScale* colorScale;
    QCPMarginGroup* marginGroup;
};

#endif // SIMPLEVISUALIZER_H
