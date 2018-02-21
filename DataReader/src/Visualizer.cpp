#include <DataReader/include/Visualizer.hpp>
#include <string>
#include <cmath>

Visualizer::Visualizer(Ui::DataReader* _ui, QCPColorMap* _colorMap, QCPColorScale* _colorScale)
{
    ui = _ui;
    colorMap = _colorMap;
    colorScale = _colorScale;
    initialized = false;
}

Visualizer::~Visualizer()
{

}

bool Visualizer::readReportFile(QString path)
{
    initialized = false;
    file.open(path.toStdString().c_str(), std::fstream::in);
    if(file.is_open() == false)
        return false;
    try {
        readEnvironment();
        initialized = true;
        renderFrame(0);
    } catch(...) {
        file.close();
        return false;
    }
    file.close();
    return true;
}

void Visualizer::renderFrame(size_t t)
{
    if(initialized != true)
        return;
    updateColorMap(t);
}

QSize Visualizer::getWindowSize()
{
    int xoffcet = 190;
    int yoffcet = 40;
    size_t XSize = (size_t)(800 * sqrt((double)X_MAX / Y_MAX));
    size_t YSize = (size_t)(800 * sqrt((double)Y_MAX / X_MAX));
    return QSize(XSize + xoffcet, YSize + yoffcet);
}

size_t Visualizer::getTotalSteps() const
{
    return totalSteps;
}

void Visualizer::removeComboBoxItems()
{
    for(int i = params-1; i >= 0; --i)
        ui->comboBox->removeItem(i);
}

void Visualizer::setProgress(double val)
{
    ui->progress->setText(QString::number(val) + "%");
    ui->progressBar->setValue(ceil(val));
}

bool Visualizer::readEnvironment()
{
    file >> MPI_NODES_X >> MPI_NODES_Y >> CUDA_X_THREADS >> CUDA_Y_THREADS
         >> TAU >> TOTAL_TIME >> STEP_LENGTH >> N_X >> N_Y
         >> X_MAX >> Y_MAX >> params;
    ui->mpi_nodes_x_2->setText(QString::number(MPI_NODES_X));
    ui->mpi_nodes_y_2->setText(QString::number(MPI_NODES_Y));
    ui->cuda_x_threads_2->setText(QString::number(CUDA_X_THREADS));
    ui->cuda_y_threads_2->setText(QString::number(CUDA_Y_THREADS));
    ui->tau_2->setText(QString::number((TAU)));
    ui->total_time_2->setText(QString::number((TOTAL_TIME)));
    ui->step_length_2->setText(QString::number((STEP_LENGTH)));
    ui->n_x_2->setText(QString::number((N_X)));
    ui->n_y_2->setText(QString::number((N_Y)));
    ui->x_max_2->setText(QString::number((X_MAX)));
    ui->y_max_2->setText(QString::number((Y_MAX)));
    totalSteps = (size_t)ceil((float)TOTAL_TIME / (float)STEP_LENGTH / TAU) - 1;
    ui->timeSlider->setMaximum(totalSteps-1);
    colorMap->data()->setSize(N_X, N_Y);
    colorMap->data()->setRange(QCPRange(0, X_MAX), QCPRange(0, Y_MAX));
    ui->customPlot->rescaleAxes();
    std::string property;
    int numberOfVariables;
    for(size_t i = 0; i < params; ++i) {
        file >> property >> numberOfVariables;
        ui->comboBox->addItem(QString::fromStdString(property));
    }
    ui->comboBox->setCurrentIndex(0);
    size_t fieldSize = N_X * N_Y;
    Field.resize(totalSteps, std::vector<std::vector<double>>());
    for(size_t s = 0; s < totalSteps; ++s) {
        Field[s].resize(params, std::vector<double>());
        for(size_t p = 0; p < params; ++p) {
            Field[s][p].resize(fieldSize, 0.0);
         }
    }
    double coef = 100.0 / (totalSteps * (double)params);
    double counter = 0.0;
    for(size_t s = 0; s < totalSteps; ++s) {
        for(size_t p = 0; p < params; ++p) {
            for(size_t i = 0; i < fieldSize; ++i) {
                file >> Field[s][p][i];
            }
            counter += 1;
            setProgress(counter * coef);
            QApplication::instance()->processEvents();
         }
    }
    setProgress(100);
    return true;
}

void Visualizer::updateColorMap(size_t t)
{
    size_t p = (size_t)ui->comboBox->currentIndex();
    size_t globalLocal;
    for(size_t x = 0; x < N_X; ++x) {
        for(size_t y = 0; y < N_Y; ++y) {
            globalLocal = y * N_X + x;
            colorMap->data()->setCell(x, y, Field[t][p][globalLocal]);
        }
    }
    colorMap->rescaleDataRange(true);
    ui->customPlot->replot();
}
