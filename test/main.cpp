#include <iostream>
using namespace std;

struct S {
    double d1;
    double d2;
    double d3;
    double d4;
};

void printS(S* s) {
    cout << "d1: " << s->d1 << "  |  d2: " << s->d2
        << "  |  d3: " << s->d3 << "  |  d4: " << s->d4 << endl;
}

int main()
{
    S* s1 = new S;
    void* v1 = (char*)s1;
    char* c1 = (char*)v1;
    S* s2 = new S;
    void* v2 = (char*)s2;
    char* c2 = (char*)v2;
    s1->d1 = 19424.125445;
    s1->d2 = 0.982374658324;
    s1->d3 = 34.5235;
    s1->d4 = 123.321;
    printS(s1);

    size_t bytes = sizeof(double);
    size_t elems = static_cast<size_t>(sizeof(S) / bytes);
    for(size_t e = 0; e < elems; ++e) {
        for(size_t b = 0; b < bytes; ++b) {
            c2[e*bytes+b] = c1[e*bytes+b];
        }
    }
    printS(s2);
    return 0;
}
