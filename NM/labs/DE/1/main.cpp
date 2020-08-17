#include <iostream>
#include <functional>

#include "../../include/de.h"

using namespace std;

double ref_func(double x){
    return (exp(x) + exp(-x) -1.0) * exp(x*x);
}

double one(const vector<double> &){
    return 1.0;
}

double z(const vector<double> &v){
    double z = v[1];
    return z;
}

double f(const vector<double> &v){
    double x = v[0];
    double z = v[1];
    double y = v[2];

    return 4.0 * x * z - (4.0 * x * x - 3.0) * y + exp(x * x);
}

int main() {
    double y0, z0;
    double a, b, h;

    string mode;

    cin >> mode;
    cin >> y0 >> z0 >> a >> b >> h;

    using func_type = function<double(const vector<double> &)>;

    vector<func_type> funcs{one, f, z};

    vector<double> p0{a, z0, y0};

    double step = h;
    double hh = h * 0.5;

    if (mode == "euler") {
        cout << "euler method" << endl;
        SODE_Euler<double, func_type> e1(funcs, p0, a, b, h);
        SODE_Euler<double, func_type> e2(funcs, p0, a, b, hh);

        for (double i = a; i <= b; i += step)
            cout << "(" << i << ";" << e1(2, i) << ")";
        cout << endl;

        double s1 = e1(2, b);
        double s2 = e2(2, b);

        cout << "error estimation: " << error_estimation(s1, s2, SODEMethod::Euler) << endl;
        cout << "absolute error: " << fabs(ref_func(b) - s1) << endl;
    } else if (mode == "runge-kutta") {
        cout << "runge-kutta method" << endl;
        SODE_RungeKutta<double, func_type> e1(funcs, p0, a, b, h);
        SODE_RungeKutta<double, func_type> e2(funcs, p0, a, b, hh);

        for (double i = a; i <= b; i += step)
            cout << "(" << i << ";" << e1(2, i) << ")";
        cout << endl;

        double s1 = e1(2, b);
        double s2 = e2(2, b);

        cout << "error estimation: " << error_estimation(s1, s2, SODEMethod::RungeKutta) << endl;
        cout << "absolute error: " << fabs(ref_func(b) - s1) << endl;
    } else if (mode == "adams") {
        cout << "adams method" << endl;
        SODE_Adams<double, func_type> e1(funcs, p0, a, b, h);
        SODE_Adams<double, func_type> e2(funcs, p0, a, b, hh);

        for (double i = a; i <= b; i += step)
            cout << "(" << i << ";" << e1(2, i) << ")";
        cout << endl;

        double s1 = e1(2, b);
        double s2 = e2(2, b);

        cout << "error estimation: " << error_estimation(s1, s2, SODEMethod::Adams) << endl;
        cout << "absolute error: " << fabs(ref_func(b) - s1) << endl;
    } else {
        cout << "There is no \"" << mode << "\" method" << endl;
        cout << "Recheck your input!" << endl;
        return 0;
    }
}
