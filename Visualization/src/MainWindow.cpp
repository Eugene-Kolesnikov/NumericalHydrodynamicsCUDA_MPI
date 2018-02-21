#include <Visualization/include/MainWindow.hpp>
#include "ui_mainwindow.h"
#include <cmath>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window | Qt::WindowTitleHint | Qt::CustomizeWindowHint);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::setWindowSize(size_t WSize, size_t HSize)
{
    int xoffcet = 190;
    int yoffcet = 40;
    size_t XSize = (size_t)(800 * sqrt((double)WSize / HSize));
    size_t YSize = (size_t)(800 * sqrt((double)HSize / WSize));
    this->setFixedSize(XSize + xoffcet, YSize + yoffcet);
}

void MainWindow::setProgress(double val)
{
    ui->progress_lbl->setText(QString::number(val) + "%");
    ui->progressBar->setValue((int)ceil(val));
}

QCustomPlot* MainWindow::getCustomPlot() const
{
    return ui->customPlot;
}

void MainWindow::setEnvironment(size_t MPI_NODES_X,
  size_t MPI_NODES_Y, size_t CUDA_X_THREADS, size_t CUDA_Y_THREADS,
  double TAU, double TOTAL_TIME, double STEP_LENGTH, size_t N_X, size_t N_Y,
  size_t X_MAX, size_t Y_MAX)
{
    ui->mpi_nodes_x->setText(QString::number(MPI_NODES_X));
    ui->mpi_nodes_y->setText(QString::number(MPI_NODES_Y));
    ui->cuda_x_threads->setText(QString::number(CUDA_X_THREADS));
    ui->cuda_y_threads->setText(QString::number(CUDA_Y_THREADS));
    ui->tau->setText(QString::number(TAU));
    ui->total_time->setText(QString::number(TOTAL_TIME));
    ui->step_length->setText(QString::number(STEP_LENGTH));
    ui->n_x->setText(QString::number(N_X));
    ui->n_y->setText(QString::number(N_Y));
    ui->x_max->setText(QString::number(X_MAX));
    ui->y_max->setText(QString::number(Y_MAX));
}
