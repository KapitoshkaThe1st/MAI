#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include "../../include/de.h"
#include "../../include/eps.h"
#include "../../include/utils.h"

using namespace std;

double t(double x) {
    return (2.0 * x + 1.0);
}

double p(double x) {
    return 4.0 * x;
}

double q(double) {
    return -4.0;
}

double w(double) {
    return 0.0;
}

double ref_func(double x) {
    return 3.0 * x + exp(-2.0 * x);
}

double one(const vector<double> &) {
    return 1.0;
}

double z(const vector<double> &v) {
    double z = v[1];
    return z;
}

double f(const vector<double> &v) {
    double x = v[0];
    double z = v[1];
    double y = v[2];

    return 4.0 * (y - x * z) / (2.0 * x + 1.0);
}

#include "../../include/deriv.h"

template <class F>
double check(F &f, double x) {
    const double d = 0.001;

    double yxx = simple_deriv(f, x, d, 2);
    double yx = simple_deriv(f, x, d);
    double y = f(x);

    return (2.0 * x + 1.0) * yxx + 4.0 * x * yx - 4.0 * y;
}

int main() {
    string mode;
    cin >> mode;

    double a0, b0, c0, a1, b1, c1, a, b, h;
    cin >> a0 >> b0 >> c0;
    cin >> a1 >> b1 >> c1;
    cin >> a >> b;
    cin >> h;

    cout << "INPUT:" << endl;
    cout << "a0:" << a0 << " b0:" << b0 << " c0:" << c0 << endl;
    cout << "a1:" << a1 << " b1:" << b1 << " c1:" << c1 << endl;
    cout << "a:" << a << " b:" << b << endl;

    using func_v_type = function<double(const vector<double> &)>;
    using func_type = function<double(double)>;

    vector<func_v_type> funcs{one, f, z};

    double step = 0.05;
    double eps = 0.0001;

    if (mode == "shooting") {
        cout << "shooting method" << endl;
        BVP_ShootingMethod<double, func_v_type> e1(funcs, a0, b0, c0, a1, b1, c1, a, b, h, eps);

        cout << "POINTS: " << endl;
        for (double i = a; i <= b; i += step) {
            cout << "(" << i << ';' << e1(i) << ")";
        }
        cout << endl;

        cout << "CHECKOUT:" << endl;
        for (double i = a + step; i <= b - step; i += step) {
            cout << check(e1, i) << ' ';
        }
        cout << endl;
    } else if (mode == "finite-difference") {
        cout << "finite-difference method" << endl;
        BVP_FiniteDifference<double, func_type> e1(t, p, q, w, a0, b0, c0, a1, b1, c1, a, b, h);

        cout << "POINTS: " << endl;
        for (double i = a; i <= b; i += step) {
            cout << "(" << i << ';' << e1(i) << ")";
        }
        cout << endl;

        cout << "CHECKOUT:" << endl;
        for (double i = a + step; i <= b - step; i += step) {
            cout << check(e1, i) << ' ';
        }

        cout << endl;
    } else {
        cout << "There is no \"" << mode << "\" method" << endl;
        cout << "Recheck your input!" << endl;
        return 0;
    }
}