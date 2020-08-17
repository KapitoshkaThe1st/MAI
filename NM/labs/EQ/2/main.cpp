#include <iostream>
#include <functional>

#include "../../include/snle_iter_methods.h"

using namespace std;

double func1(const vector<double> &v){
    double x = v[0], y = v[1];
    return x*x+y*y-9.0;
}

double func2(const vector<double> &v){
    double x = v[0], y = v[1];
    return x-exp(y)+3;
}

double iter_func1(const vector<double> &v){
    double y = v[1];
    return sqrt(9.0-y*y);
}

double iter_func2(const vector<double> &v){
    double x = v[0];
    return log(x+3.0);
}

int main(){

    double prec = 0.00001;

    vector<double> x_0;

    string mode;
    cin >> mode;

    double precision;
    cin >> precision;

    double a, b, c, d;
    cin >> a >> b >> c >> d;

    if(mode == "newton"){
        cout << "newton method" << endl;

        vector<double> init{(a+b)/2.0, (c+d)/2.0};
        cout << "initial approximation: " << init << endl;
        x_0 = SNLE::newton_method(vector<function<double(vector<double>)>>({func1, func2}), init, prec);
    }
    else if(mode == "fixed-point"){
        cout << "fixed-point method" << endl;

        vector<function<double(vector<double>)>> vf{iter_func1, iter_func2};
        double q = numeric_limits<double>::min();
        vector<double> p{a, c};
        Matrix<double> j(2);

        double grid_check_step = 0.01;

        for(; p[0] < b; p[0] += grid_check_step)
            for(; p[1] < d; p[1] += grid_check_step){
                jacobi_matrix(vf, p, precision, j);
                double norm = j.norm_2();
                if(q < norm)
                    q = norm;
            }

        cout << "q: " << q << endl;
        if(q < 1.0){
            vector<double> init{(a+b)/2.0, (c+d)/2.0};
            cout << "initial approximation: " << init << endl;
            x_0 = SNLE::fixed_point_iter(vf, init, q, prec);
        }
        else{
            cout << "neccessary condition q < 1 is not met" << endl;
            return 0;
        }
    }
    else{
        cout << "There is no \"" << mode << "\" method" << endl;
        cout << "Recheck your input!" << endl;
        return 0;
    }

    cout << "x_0: " << x_0 << endl;

    cout << "iterations required: " << SNLE::iters_required() << endl;

    cout << "-- CHECKOUT --" << endl;
    cout << "func1(x_0): " << func1(x_0) << endl;
    cout << "func2(x_0): " <<  func2(x_0) << endl;
}