#include <iostream>
#include <vector>

#include "../../include/matrix.h"

using namespace std;

int main(){
    cout << "--- INPUT ---\n";

    Matrix<double> a(cin);

    cout << "A:\n" << a << endl; 

    LU<double> lu(a);

    int n = lu.size();
    vector<double> b(n);

    cout << "b: ";
    for(double &i : b){
        cin >> i;
        cout << i << ' ';
    }
    cout << endl;

    cout << "\n--- SOLUTION ---\n";

    vector<double> x = lu.slae_solve(b);

    cout << "x: ";
    for(double i : x)
        cout << i << ' ';
    cout << endl;

    cout << "det: " << lu.det() << endl;

    Matrix<double> inverse = lu.inverse();
    cout << "inverse matrix: \n" << inverse << endl;

    cout << "\n--- CHECKOUT ---\n";

    cout << "A*x:\n";
    for(double i : a * x)
        cout << i << ' ';
    cout << endl;

    cout << "A*A^(-1):\n" << a * inverse << endl;
}