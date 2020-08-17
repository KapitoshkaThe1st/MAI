#include <iostream>

#include "../../include/matrix.h"
#include "../../include/eigenv.h"

using namespace std;

int main(){
    cout << "-- INPUT --" << endl;
    Matrix<double> m(cin);

    cout << "M:" << endl;
    cout << m << endl;

    double precision;
    cin >> precision;

    cout << "-- SOLUTION --" << endl;

    cout << "jacobi method" << endl;
    cout << "with precision: " << precision << endl;

    vector<double> eigen_values;
    vector<vector<double>> eigen_vectors;

    EigenV::jacobi(m, eigen_values, eigen_vectors, precision);

    cout << "iterations required: " << EigenV::iters_required() << endl;

    cout << "  e-val\t\t\te-vec" << endl;
    int count = eigen_values.size();
    for(int i = 0; i < count; ++i){
        cout << eigen_values[i] << ": ( " << eigen_vectors[i] << ")" << endl;
    }

    cout << "-- CHECKOUT --" << endl;
    for(int i = 0; i < count; ++i){
        cout << "    M * e-vec: " << m * eigen_vectors[i] << endl;
        cout << "e-val * e-vec: " << eigen_values[i] * eigen_vectors[i] << endl;
    }
}
