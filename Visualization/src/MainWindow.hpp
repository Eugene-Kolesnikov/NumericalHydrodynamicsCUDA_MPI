#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "Qcustomplot.hpp"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

public:
    void setWindowSize(size_t WSize, size_t HSize);
    void setProgress(double val);
    QCustomPlot* getCustomPlot() const;
    void setEnvironment(size_t MPI_NODES_X,
      size_t MPI_NODES_Y, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS,
      double TAU, double TOTAL_TIME, double STEP_LENGTH, size_t N_X, size_t N_Y,
      size_t X_MAX, size_t Y_MAX);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
