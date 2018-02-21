#ifndef DATAREADER_HPP
#define DATAREADER_HPP

#include <QMainWindow>
#include <utilities/QCustomPlot/include/Qcustomplot.hpp>
#include <DataReader/include/Visualizer.hpp>

namespace Ui {
class DataReader;
}

class Visualizer;

class DataReader : public QMainWindow
{
    Q_OBJECT

public:
    explicit DataReader(QWidget *parent = 0);
    ~DataReader();

protected:
    void initCustomPlot();

private slots:
    void on_open_btn_clicked();
    void on_browse_btn_clicked();

    void on_anthr_report_btn_clicked();

    void on_timeSlider_sliderMoved(int position);

    void on_comboBox_currentIndexChanged(int index);

    void on_timeSlider_valueChanged(int value);

protected:
    QString reportPath;

private:
    Ui::DataReader* ui;
    Visualizer* visualizer;

protected:
    QCPColorMap* colorMap;
    QCPColorScale* colorScale;
    QCPMarginGroup* marginGroup;
};

#endif // DATAREADER_HPP
