#include <iostream>
#include <cmath>
#include <cstring>

#include "../../include/nle_iter_methods.h"
#include "../../include/deriv.h"

using namespace std;

double func(double x){
    return log(x+1.0)-2*x*x+1;
}

double iter_func(double x){
    return sqrt((1.0+log(x+1.0))/2.0);
}

int main(int argc, char **argv){
    if(argc < 5){
        cout << "usage: fixed-point|newton|dichotomy precision left_bound right_bound" << endl;
        return 0;
    }

    cout << "-- SOLUTION --" << endl;

    double precision = stod(argv[2]);

    double x_0 = 0.0;
    if(!strcmp(argv[1], "fixed-point")){
        cout << "fixed-point iteration method" << endl;

        double a = stod(argv[3]), b = stod(argv[4]);

        double q = numeric_limits<double>::min();
        for(double x = a; x < b; x += precision){
            double dfx = fabs(simple_deriv(iter_func, x, precision, 1));
            if(dfx > q)
                q = dfx;
        }
        cout << "q: " << q << endl;
        if(q < 1.0){
            double init = (a+b)*0.5;
            cout << "initial approximation: " << init << endl;

            x_0 = NLE::fixed_point_iter(iter_func, init, q, precision);
        }
        else{
            cout << "neccessary condition q < 1 is not met" << endl;
            return 0;
        }
    }
    else if(!strcmp(argv[1], "newton")){
        cout << "newton method" << endl;

        double a = stod(argv[3]), b = stod(argv[4]);

        double a_criteria = simple_deriv(func, a, precision, 2) * func(a);
        double b_criteria = simple_deriv(func, b, precision, 2) * func(b);

        if(a_criteria > 0.0){
            cout << "initial approximation: " << a << endl;
            x_0 = NLE::newton_method(func, a, precision);
        }
        else if(b_criteria > 0.0){
            cout << "initial approximation: " << b << endl;
            x_0 = NLE::newton_method(func, b, precision);
        }
        else{
            cout << "bad [a,b] range! Criteria d2f/dx2 * f > 0 is not met" << endl;
            return 0;
        }
    }
    else if(!strcmp(argv[1], "dichotomy")){
        cout << "dichotomy method" << endl;

        double a = stod(argv[3]), b = stod(argv[4]);
        x_0 = NLE::dichotomy_method(func, a, b, precision);
    }
    else{
        cout << "There is no \"" << argv[1] << "\" method" << endl;
        cout << "Recheck your input!" << endl;
        return 0;
    }

    cout << "with precision: " << precision << endl;
    cout << "iterations required: " << NLE::iters_required() << endl;

    cout << "x_0: " << x_0 << endl; 

    cout << "-- CHECKOUT --" << endl;
    cout << "f(x_0): " << func(x_0) << endl;
}
