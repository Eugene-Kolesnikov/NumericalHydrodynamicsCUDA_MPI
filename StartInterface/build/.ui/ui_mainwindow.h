/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.10.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QGroupBox *groupBox;
    QLabel *label;
    QLineEdit *mpi_nodes_x;
    QLabel *label_2;
    QLineEdit *mpi_nodes_y;
    QLabel *label_3;
    QLabel *label_4;
    QGroupBox *groupBox_2;
    QLabel *label_5;
    QLineEdit *cuda_x_threads;
    QLabel *label_6;
    QLineEdit *cuda_y_threads;
    QLabel *label_7;
    QLabel *label_8;
    QGroupBox *groupBox_3;
    QLabel *label_9;
    QLineEdit *tau;
    QLabel *label_10;
    QLineEdit *total_time;
    QLabel *label_11;
    QLabel *label_12;
    QLabel *label_13;
    QLineEdit *step_length;
    QLabel *label_14;
    QLabel *label_15;
    QLabel *label_16;
    QLineEdit *n_x;
    QLabel *label_17;
    QLabel *label_18;
    QLineEdit *n_y;
    QGroupBox *groupBox_4;
    QRadioButton *radioButton;
    QRadioButton *radioButton_2;
    QGroupBox *groupBox_5;
    QRadioButton *radioButton_3;
    QRadioButton *radioButton_4;
    QRadioButton *radioButton_5;
    QPushButton *startbtn;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(981, 500);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QStringLiteral("centralwidget"));
        groupBox = new QGroupBox(centralwidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        groupBox->setGeometry(QRect(30, 20, 661, 101));
        label = new QLabel(groupBox);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(20, 30, 91, 25));
        mpi_nodes_x = new QLineEdit(groupBox);
        mpi_nodes_x->setObjectName(QStringLiteral("mpi_nodes_x"));
        mpi_nodes_x->setGeometry(QRect(120, 30, 41, 25));
        mpi_nodes_x->setAlignment(Qt::AlignCenter);
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(170, 30, 351, 25));
        mpi_nodes_y = new QLineEdit(groupBox);
        mpi_nodes_y->setObjectName(QStringLiteral("mpi_nodes_y"));
        mpi_nodes_y->setGeometry(QRect(120, 60, 41, 25));
        mpi_nodes_y->setAlignment(Qt::AlignCenter);
        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(20, 60, 91, 25));
        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(170, 60, 351, 25));
        groupBox_2 = new QGroupBox(centralwidget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        groupBox_2->setGeometry(QRect(30, 140, 661, 101));
        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QStringLiteral("label_5"));
        label_5->setGeometry(QRect(20, 30, 121, 25));
        cuda_x_threads = new QLineEdit(groupBox_2);
        cuda_x_threads->setObjectName(QStringLiteral("cuda_x_threads"));
        cuda_x_threads->setGeometry(QRect(150, 30, 41, 25));
        cuda_x_threads->setAlignment(Qt::AlignCenter);
        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QStringLiteral("label_6"));
        label_6->setGeometry(QRect(200, 30, 361, 25));
        cuda_y_threads = new QLineEdit(groupBox_2);
        cuda_y_threads->setObjectName(QStringLiteral("cuda_y_threads"));
        cuda_y_threads->setGeometry(QRect(150, 60, 41, 25));
        cuda_y_threads->setAlignment(Qt::AlignCenter);
        label_7 = new QLabel(groupBox_2);
        label_7->setObjectName(QStringLiteral("label_7"));
        label_7->setGeometry(QRect(20, 60, 121, 25));
        label_8 = new QLabel(groupBox_2);
        label_8->setObjectName(QStringLiteral("label_8"));
        label_8->setGeometry(QRect(200, 60, 361, 25));
        groupBox_3 = new QGroupBox(centralwidget);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        groupBox_3->setGeometry(QRect(30, 260, 661, 191));
        label_9 = new QLabel(groupBox_3);
        label_9->setObjectName(QStringLiteral("label_9"));
        label_9->setGeometry(QRect(20, 30, 121, 25));
        tau = new QLineEdit(groupBox_3);
        tau->setObjectName(QStringLiteral("tau"));
        tau->setGeometry(QRect(60, 30, 71, 25));
        tau->setAlignment(Qt::AlignCenter);
        label_10 = new QLabel(groupBox_3);
        label_10->setObjectName(QStringLiteral("label_10"));
        label_10->setGeometry(QRect(140, 30, 71, 25));
        total_time = new QLineEdit(groupBox_3);
        total_time->setObjectName(QStringLiteral("total_time"));
        total_time->setGeometry(QRect(110, 60, 61, 25));
        total_time->setAlignment(Qt::AlignCenter);
        label_11 = new QLabel(groupBox_3);
        label_11->setObjectName(QStringLiteral("label_11"));
        label_11->setGeometry(QRect(20, 60, 81, 25));
        label_12 = new QLabel(groupBox_3);
        label_12->setObjectName(QStringLiteral("label_12"));
        label_12->setGeometry(QRect(180, 60, 301, 25));
        label_13 = new QLabel(groupBox_3);
        label_13->setObjectName(QStringLiteral("label_13"));
        label_13->setGeometry(QRect(20, 90, 101, 25));
        step_length = new QLineEdit(groupBox_3);
        step_length->setObjectName(QStringLiteral("step_length"));
        step_length->setGeometry(QRect(120, 90, 51, 25));
        step_length->setAlignment(Qt::AlignCenter);
        label_14 = new QLabel(groupBox_3);
        label_14->setObjectName(QStringLiteral("label_14"));
        label_14->setGeometry(QRect(180, 90, 461, 25));
        label_15 = new QLabel(groupBox_3);
        label_15->setObjectName(QStringLiteral("label_15"));
        label_15->setGeometry(QRect(150, 120, 291, 25));
        label_16 = new QLabel(groupBox_3);
        label_16->setObjectName(QStringLiteral("label_16"));
        label_16->setGeometry(QRect(20, 120, 31, 25));
        n_x = new QLineEdit(groupBox_3);
        n_x->setObjectName(QStringLiteral("n_x"));
        n_x->setGeometry(QRect(60, 120, 81, 25));
        n_x->setAlignment(Qt::AlignCenter);
        label_17 = new QLabel(groupBox_3);
        label_17->setObjectName(QStringLiteral("label_17"));
        label_17->setGeometry(QRect(20, 150, 31, 25));
        label_18 = new QLabel(groupBox_3);
        label_18->setObjectName(QStringLiteral("label_18"));
        label_18->setGeometry(QRect(150, 150, 291, 25));
        n_y = new QLineEdit(groupBox_3);
        n_y->setObjectName(QStringLiteral("n_y"));
        n_y->setGeometry(QRect(60, 150, 81, 25));
        n_y->setAlignment(Qt::AlignCenter);
        groupBox_4 = new QGroupBox(centralwidget);
        groupBox_4->setObjectName(QStringLiteral("groupBox_4"));
        groupBox_4->setGeometry(QRect(720, 20, 231, 101));
        radioButton = new QRadioButton(groupBox_4);
        radioButton->setObjectName(QStringLiteral("radioButton"));
        radioButton->setGeometry(QRect(20, 30, 191, 25));
        radioButton->setChecked(true);
        radioButton_2 = new QRadioButton(groupBox_4);
        radioButton_2->setObjectName(QStringLiteral("radioButton_2"));
        radioButton_2->setEnabled(false);
        radioButton_2->setGeometry(QRect(20, 60, 181, 25));
        radioButton_2->setCheckable(false);
        radioButton_2->setAutoExclusive(true);
        groupBox_5 = new QGroupBox(centralwidget);
        groupBox_5->setObjectName(QStringLiteral("groupBox_5"));
        groupBox_5->setGeometry(QRect(720, 140, 231, 131));
        radioButton_3 = new QRadioButton(groupBox_5);
        radioButton_3->setObjectName(QStringLiteral("radioButton_3"));
        radioButton_3->setGeometry(QRect(20, 30, 191, 25));
        radioButton_3->setChecked(true);
        radioButton_4 = new QRadioButton(groupBox_5);
        radioButton_4->setObjectName(QStringLiteral("radioButton_4"));
        radioButton_4->setEnabled(false);
        radioButton_4->setGeometry(QRect(20, 60, 181, 25));
        radioButton_4->setCheckable(false);
        radioButton_4->setAutoExclusive(true);
        radioButton_5 = new QRadioButton(groupBox_5);
        radioButton_5->setObjectName(QStringLiteral("radioButton_5"));
        radioButton_5->setEnabled(false);
        radioButton_5->setGeometry(QRect(20, 90, 181, 25));
        radioButton_5->setCheckable(false);
        radioButton_5->setAutoExclusive(true);
        startbtn = new QPushButton(centralwidget);
        startbtn->setObjectName(QStringLiteral("startbtn"));
        startbtn->setGeometry(QRect(720, 290, 231, 61));
        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName(QStringLiteral("menubar"));
        menubar->setGeometry(QRect(0, 0, 981, 22));
        MainWindow->setMenuBar(menubar);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QStringLiteral("statusbar"));
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "Main Interface", nullptr));
        groupBox->setTitle(QApplication::translate("MainWindow", "MPI Configuration:", nullptr));
        label->setText(QApplication::translate("MainWindow", "MPI_NODES_X:", nullptr));
        mpi_nodes_x->setText(QApplication::translate("MainWindow", "1", nullptr));
        label_2->setText(QApplication::translate("MainWindow", "(amount of computational nodes along the X-direction)", nullptr));
        mpi_nodes_y->setText(QApplication::translate("MainWindow", "1", nullptr));
        label_3->setText(QApplication::translate("MainWindow", "MPI_NODES_Y:", nullptr));
        label_4->setText(QApplication::translate("MainWindow", "(amount of computational nodes along the Y-direction)", nullptr));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "GPU Configuration:", nullptr));
        label_5->setText(QApplication::translate("MainWindow", "CUDA_X_THREADS:", nullptr));
        cuda_x_threads->setText(QApplication::translate("MainWindow", "32", nullptr));
        label_6->setText(QApplication::translate("MainWindow", "(amount of threads in a CUDA block along the X-direction)", nullptr));
        cuda_y_threads->setText(QApplication::translate("MainWindow", "32", nullptr));
        label_7->setText(QApplication::translate("MainWindow", "CUDA_Y_THREADS:", nullptr));
        label_8->setText(QApplication::translate("MainWindow", "(amount of threads in a CUDA block along the Y-direction)", nullptr));
        groupBox_3->setTitle(QApplication::translate("MainWindow", "Solver Configuration:", nullptr));
        label_9->setText(QApplication::translate("MainWindow", "TAU:", nullptr));
        tau->setText(QApplication::translate("MainWindow", "1e-5", nullptr));
        label_10->setText(QApplication::translate("MainWindow", "(time step)", nullptr));
        total_time->setText(QApplication::translate("MainWindow", "5", nullptr));
        label_11->setText(QApplication::translate("MainWindow", "TOTAL_TIME:", nullptr));
        label_12->setText(QApplication::translate("MainWindow", "(time from 0 to TOTAL_TIME with the step TAU)", nullptr));
        label_13->setText(QApplication::translate("MainWindow", "STEP_LENGTH:", nullptr));
        step_length->setText(QApplication::translate("MainWindow", "1", nullptr));
        label_14->setText(QApplication::translate("MainWindow", "(number of loop steps which must be skipped before each visualization call)", nullptr));
        label_15->setText(QApplication::translate("MainWindow", "(discretization of the grid along the X-direction)", nullptr));
        label_16->setText(QApplication::translate("MainWindow", "N_X:", nullptr));
        n_x->setText(QApplication::translate("MainWindow", "100", nullptr));
        label_17->setText(QApplication::translate("MainWindow", "N_Y:", nullptr));
        label_18->setText(QApplication::translate("MainWindow", "(discretization of the grid along the Y-direction)", nullptr));
        n_y->setText(QApplication::translate("MainWindow", "100", nullptr));
        groupBox_4->setTitle(QApplication::translate("MainWindow", "Computational Model:", nullptr));
        radioButton->setText(QApplication::translate("MainWindow", " Lattice Boltzmann Method", nullptr));
        radioButton_2->setText(QApplication::translate("MainWindow", " Navier-Stokes", nullptr));
        groupBox_5->setTitle(QApplication::translate("MainWindow", "Computational Model:", nullptr));
        radioButton_3->setText(QApplication::translate("MainWindow", " Uniform square grid", nullptr));
        radioButton_4->setText(QApplication::translate("MainWindow", " Stratified square grid", nullptr));
        radioButton_5->setText(QApplication::translate("MainWindow", " Random triangular grid", nullptr));
        startbtn->setText(QApplication::translate("MainWindow", "Start the simulation", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
