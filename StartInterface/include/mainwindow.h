#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <utilities/libLoader/include/libLoader.hpp>
#include <utilities/Register/SystemRegister.hpp>
#include <string>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void connectParserLib();
    void readPrevConfig();

private slots:
    void startSimulation();

private:
    Ui::MainWindow *ui;
    DLHandler parserLibHandler;
    std::string CaseConfigFile;
};

#endif // MAINWINDOW_H
