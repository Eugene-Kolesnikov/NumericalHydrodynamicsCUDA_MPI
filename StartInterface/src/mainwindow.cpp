#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <list>
#include <map>
#include <string>
#include <cstdio>
#include <exception>
#include <QDir>
#include <dlfcn.h>
#include <cstdlib>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->startbtn, SIGNAL(released()),this, SLOT(startSimulation()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::connectParserLib()
{
    std::string libpath = QCoreApplication::applicationDirPath().toStdString() + "/libConfigParser.1.0.0.dylib";
    parserLibHandle = dlopen(libpath.c_str(),
                             RTLD_LOCAL | RTLD_LAZY);
    if (!parserLibHandle) {
        throw std::runtime_error(dlerror());
    } else {
        printf("Opened the parser dynamic library.\n");
    }
    createConfig = (void (*)(void*, const char*))dlsym(parserLibHandle, "createConfig");
    readConfig = (void* (*)(const char*))dlsym(parserLibHandle, "readConfig");
    if(!readConfig || !createConfig)
        throw std::runtime_error("Can't load functions from the dynamic library!");
}

void MainWindow::readPrevConfig()
{
    using namespace std;
    string filepath = QCoreApplication::applicationDirPath().toStdString() + "/CONFIG";
    void* lst = readConfig(filepath.c_str());
    if(lst == nullptr)
        return;
    list<pair<string,double>>* params = (list<pair<string,double>>*)lst;
    auto it = params->begin();
    ui->mpi_nodes_x->setText(QString::number((it++)->second));
    ui->mpi_nodes_y->setText(QString::number((it++)->second));
    ui->cuda_x_threads->setText(QString::number((it++)->second));
    ui->cuda_y_threads->setText(QString::number((it++)->second));
    ui->tau->setText(QString::number((it++)->second));
    ui->total_time->setText(QString::number((it++)->second));
    ui->step_length->setText(QString::number((it++)->second));
    ui->n_x->setText(QString::number((it++)->second));
    ui->n_y->setText(QString::number((it++)->second));
    delete params;
}

void MainWindow::startSimulation()
{
    ui->startbtn->setEnabled(false);
    using namespace std;
    list<pair<string,double>> params;
    params.push_back(make_pair<string,double>("MPI_NODES_X",ui->mpi_nodes_x->text().toDouble()));
    params.push_back(make_pair<string,double>("MPI_NODES_Y",ui->mpi_nodes_y->text().toDouble()));
    params.push_back(make_pair<string,double>("CUDA_X_THREADS",ui->cuda_x_threads->text().toDouble()));
    params.push_back(make_pair<string,double>("CUDA_Y_THREADS",ui->cuda_y_threads->text().toDouble()));
    params.push_back(make_pair<string,double>("TAU",ui->tau->text().toDouble()));
    params.push_back(make_pair<string,double>("TOTAL_TIME",ui->total_time->text().toDouble()));
    params.push_back(make_pair<string,double>("STEP_LENGTH",ui->step_length->text().toDouble()));
    params.push_back(make_pair<string,double>("N_X",ui->n_x->text().toDouble()));
    params.push_back(make_pair<string,double>("N_Y",ui->n_y->text().toDouble()));
    params.push_back(make_pair<string,double>("LBM",1));
    params.push_back(make_pair<string,double>("NS",0));
    params.push_back(make_pair<string,double>("USG",1));
    params.push_back(make_pair<string,double>("STAG",0));
    params.push_back(make_pair<string,double>("RND_TR",0));
    try {
        createConfig((void*)&params, (filepath + "CONFIG").c_str());
    } catch(runtime_error err) {
        throw std::runtime_error(err.what());
    }
    string filepath = QCoreApplication::applicationDirPath().toStdString() + "/";
    size_t mpi_nodes = ui->mpi_nodes_x->text().toUInt() * ui->mpi_nodes_y->text().toUInt() + 1;
    string system_call = "mpiexec -l -np " + std::to_string(mpi_nodes) +
            "./simulation_app " + filepath;
    system(system_call);
    emit close();
}