#include <iostream>
#include <vector>
#include <cmath>

#include "../../include/matrix.h"
#include "../../include/eps.h"
#include "../../include/slae_iter_methods.h"

using namespace std;

int main(int argc, char **argv){
    if(argc < 2){
        cout << "usage ./prog iterative|seidel < input_file" << endl;
        return 0;
    }

    Matrix<double> m(cin);

    int n = m.size();

    cout << "--- INPUT ---\n";

    cout << "A: ";
    cout << m << endl;

    cout << "b: ";
    vector<double> b(n);
    for(double &i : b){
        cin >> i;
        cout << i << ' ';
    }
    cout << endl;

    double precision;
    cin >> precision;

    cout << "\n--- SOLUTION ---\n";
    vector<double> x;

    if(!strcmp(argv[1], "iterative")){
        cout << "iterative method" << endl;
        x = SLAE::iterative_method(m, b, precision);
    }
    else if(!strcmp(argv[1], "seidel")){
        cout << "seidel method" << endl;
        x = SLAE::seidel_method(m, b, precision);
    }
    else{
        cout << "There is no \"" << argv[1] << "\" method" << endl;
        cout << "Recheck your input!" << endl;
        return 0;
    }
    cout << "with precision: " << precision << endl;

    cout << "iterations required: " << SLAE::iters_required() << endl;

    cout << "x: ";
    for(double i : x)
        cout << i << ' ';
    cout << endl;

    cout << "\n--- CHECKOUT ---\n";

    vector<double> check = m * x;

    cout << "A*x: " << endl;
    for(double i : check)
        cout << i << ' ';
    cout << endl;
}
