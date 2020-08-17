#include <iostream>
#include <vector>
#include <tuple>

#include "../../include/eigenv.h"
#include "../../include/matrix.h"

using namespace std;

int main(){
    cout << "-- INPUT --" << endl;
    Matrix<double> m(cin);

    cout << "M:" << endl;
    cout << m << endl;

    double precision;
    cin >> precision;

    cout << "-- SOLUTION --" << endl;

    cout << "QR algorithm method" << endl;
    cout << "with precision: " << precision << endl;

    vector<complex<double>> eval = EigenV::QR_eigenV(m, precision);

    cout << "iterations required: " << EigenV::iters_required() << endl;
    cout << "e-val: " << eval << endl;
}