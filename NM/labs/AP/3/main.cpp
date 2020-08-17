#include <iostream>
#include <vector>

#include "../../include/lsm.h"

using namespace std;

int main(){
    int n;
    cin >> n;

    vector<double> x(n), f(n);

    for(int i = 0; i < n; ++i)
        cin >> x[i];

    for (int i = 0; i < n; ++i)
        cin >> f[i];

    cout << "POINTS:\n";
    for(int i = 0; i < n; ++i){
        cout << "(" << x[i] << ";" << f[i] << ")";
    }
    cout << endl;

    int m;
    cin >> m;

    LSM lsm(x, f, m+1);

    cout << "COEFS:\n";
    lsm.print_coefs();

    double err = 0.0;
    for(int i = 0; i < n; ++i){
        double d = f[i] - lsm(x[i]);
        err += d*d;
    }
    cout << "err: " << err << endl;

}