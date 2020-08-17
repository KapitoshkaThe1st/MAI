#include <iostream>
#include <vector>

#include "../../include/interpolation.h"

using namespace std;

int main(){
    int n;
    cin >> n;

    vector<double> x(n), f(n);
    for(int i = 0; i < n; ++i){
        cin >> x[i];
    }

    for(int i = 0; i < n; ++i){
        cin >> f[i];
    }

    cout << "x: " << x << endl;
    cout << "f: " << f << endl;

    CubicSpline<double> cs(x, f);

    double xx;
    cin >> xx;

    cout << "x*: " << xx << endl;

    cout << "CS(x*): " << cs(xx) << endl;

}
