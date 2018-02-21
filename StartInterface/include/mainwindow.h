#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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
    void (*createConfig)(void* params, const char* filepath);
    void* (*readConfig)(const char* filepath);
    void* parserLibHandle;
};

#endif // MAINWINDOW_H
