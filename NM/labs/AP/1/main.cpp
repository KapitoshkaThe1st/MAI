#include <iostream>
#include <cmath>
#include <string>

#include "../../include/interpolation.h"

using namespace std;

double func(double x){
    return asin(x);
    // double pi = 3.1415926535897932384626433;
    // return sin(pi*x/6.0);
}

int main(){
    string mode;
    cin >> mode;

    int n;
    cin >> n;

    cout << "x : " << "f(x)" << endl;
    vector<double> x(n);
    vector<double> f(n);
    for(int i = 0; i < n; ++i){
        cin >> x[i];
        f[i] = func(x[i]);
        cout << x[i] << " : " << f[i] << endl;
    }

    double xx;
    cin >> xx;

    cout << "\nx*: " << xx << endl;
    // cout << "x:" << endl;
    // for(auto e : x)
    //     cout << e << ' ';
    // cout << endl;

    if(mode == "lagrange"){
        LagrangePolynomial ip(x, f);

        cout << "f(x): " << func(xx) << endl;
        cout << "f(ip): " << ip(xx) << endl;

        cout << "error:" << fabs(ip(xx) - func(xx)) << endl;

        cout << "-- CHECKOUT --" << endl;
        for (int i = 0; i < n; ++i) {
            cout << "f(" << x[i] << "): " << func(x[i]) << endl;
            cout << "ip(" << x[i] << "): " << ip(x[i]) << endl;
        }
    }
    else if(mode == "newton"){
        NewtonPolynomial ip(x, f);

        cout << "f(x): " << func(xx) << endl;
        cout << "ip(x): " << ip(xx) << endl;

        cout << "error:" << fabs(ip(xx) - func(xx)) << endl;

        cout << "-- CHECKOUT --" << endl;
        for (int i = 0; i < n; ++i) {
            cout << "f(" << x[i] << "): " << func(x[i]) << endl;
            cout << "ip(" << x[i] << "): " << ip(x[i]) << endl;
        }
    }
    else {
        cout << "There is no \"" << mode << "\" method" << endl;
        cout << "Recheck your input!" << endl;
        return 0;
    }
}