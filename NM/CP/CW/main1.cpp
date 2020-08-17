#include <iostream>
#include <cstdlib>
#include <vector>

#include <cmath>
#include "../include/multdim_integral.h"

using namespace std;

double f(const std::vector<double> &v){
    double sum = 0.0;
    for(auto i : v){
        for(auto j : v){
            sum += i * j;
        }
    }
    return sum + 1.0;
}

int main(){
    srand(time(0));
    rand();

    double eps = 0.1;
    double M = 2.0;
    vector<double> ref_values{12.0, 495.0/2.0, 6129.0/2.0, 26568.0};

    cout << "eps: " << eps << endl;

    for(unsigned int k = 1; k <= 4; ++k){
        vector<double> a(k), b(k);
        for(unsigned int i = 0; i < k; ++i){
            a[i] = 2 * i;
            b[i] = 2 * i + 3;
        }

        cout << "a: ";
        for(auto i : a)
            cout << i << ' ';
        cout << endl;

        cout << "b: ";
        for(auto i : b)
            cout << i << ' ';
        cout << endl;

        cout.precision(17);

        double mc_res = monte_carlo_prec(f, a, b, eps);
        double err = abs(mc_res - ref_values[k-1]);

        cout << "ref_res: " << ref_values[k-1] << endl;

        cout << "mc_res: " << mc_res << endl;
        cout << "mc_err: " << err << endl;


        Quadrature<decltype(f)> q(f, a, b, eps, M);
        double q_res = q.integral();
        cout << "q_res: " << q_res << endl;
        cout << "q_err: " << abs(q_res - ref_values[k-1]) << endl;
        cout << endl;
    }
}