#include <DataReader/include/Datareader.hpp>
#include "ui_datareader.h"
#include <QFileDialog>
#include <QMessageBox>

DataReader::DataReader(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::DataReader)
{
    ui->setupUi(this);
    ui->progressBar->setValue(0);
    ui->stackedWidget->setCurrentIndex(0);
    setFixedSize(700, 175);
    initCustomPlot();
    visualizer = new Visualizer(ui, colorMap, colorScale);
}

DataReader::~DataReader()
{
    delete ui;
}

void DataReader::initCustomPlot()
{
    ui->customPlot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom);
    ui->customPlot->axisRect()->setupFullAxesBox(true);
    ui->customPlot->xAxis->setLabel("x");
    ui->customPlot->yAxis->setLabel("y");
    // set up the QCPColorMap:
    colorMap = new QCPColorMap(ui->customPlot->xAxis, ui->customPlot->yAxis);
    colorScale = new QCPColorScale(ui->customPlot);
    // add it to the right of the main axis rect
    ui->customPlot->plotLayout()->addElement(0, 1, colorScale);
    // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorScale->setType(QCPAxis::atRight);
    // associate the color map with the color scale
    colorMap->setColorScale(colorScale);
    // set the color gradient of the color map to one of the presets:
    colorMap->setGradient(QCPColorGradient::gpJet);
    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    marginGroup = new QCPMarginGroup(ui->customPlot);
    ui->customPlot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
}

void DataReader::on_open_btn_clicked()
{
    if(reportPath.isEmpty()) {
        QMessageBox::question(this, "Error", "Please choose the report file.", QMessageBox::Ok);
    } else {
        ui->stackedWidget->setCurrentIndex(1);
        setFixedSize(700, 170);
        bool status = visualizer->readReportFile(reportPath);
        if(status == true) {
            setFixedSize(visualizer->getWindowSize());
            ui->timeSlider->setValue(0);
            ui->time_lbl->setText(QString("1 / ") + QString::number(visualizer->getTotalSteps()));
            ui->stackedWidget->setCurrentIndex(2);
            ui->progressBar->setValue(0);
        } else {
            QMessageBox::question(this, "Error", "Was not able to read the file!", QMessageBox::Ok);
            ui->progressBar->setValue(0);
            ui->stackedWidget->setCurrentIndex(0);
            setFixedSize(700, 175);
        }
    }
}

void DataReader::on_browse_btn_clicked()
{
    reportPath = QFileDialog::getOpenFileName(this,
             tr("Open data File"), QCoreApplication::applicationDirPath() + "../..");
    ui->reppath->setText(reportPath);
}

void DataReader::on_anthr_report_btn_clicked()
{
    ui->stackedWidget->setCurrentIndex(0);
    setFixedSize(700, 175);
    visualizer->removeComboBoxItems();
}

void DataReader::on_timeSlider_sliderMoved(int position)
{
    ui->time_lbl->setText(QString::number(position+1) +
        QString(" / ") + QString::number(visualizer->getTotalSteps()));
    ui->timeSlider->setValue(position);
    visualizer->renderFrame(position);
}

void DataReader::on_comboBox_currentIndexChanged(int index)
{
    visualizer->renderFrame(ui->timeSlider->value());
}

void DataReader::on_timeSlider_valueChanged(int value)
{
    ui->time_lbl->setText(QString::number(value+1) +
        QString(" / ") + QString::number(visualizer->getTotalSteps()));
    ui->timeSlider->setValue(value);
    visualizer->renderFrame(value);
}
