#include <iostream>
#include <vector>

#include "../../include/deriv.h"

using namespace std;

int main(){
    int n;
    cin >> n;

    vector<double> x(n), f(n);

    for(int i = 0; i < n; ++i)
        cin >> x[i];

    for(int i = 0; i < n; ++i)
        cin >> f[i];


    cout << "POINTS: " << endl;
    for(int i = 0; i < n; ++i)
        cout << "(" << x[i] << ";" << f[i] << ")";
    cout << endl;

    double xx;
    cin >> xx;

    cout << "x*: " << xx << endl;

    cout << "f'(x*): " << deriv(x, f, xx) << endl;
    cout << "f''(x*): " << deriv2(x, f, xx) << endl;
}
