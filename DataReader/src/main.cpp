#include "Datareader.hpp"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    DataReader w;
    w.show();

    return a.exec();
}
