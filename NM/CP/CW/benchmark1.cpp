#include <iostream>
#include <cstdlib>
#include <vector>

#include <cmath>
#include "../include/multdim_integral.h"

using namespace std;

double f(const std::vector<double> &v){
    double sum = 0.0;

    size_t n = v.size();
    double t;
    for(size_t i = 0; i < n; ++i){
        t = v[i];
        sum += t*t;
    }

    return sum;
}

#include <chrono>

template<typename T>
void print_vec_as_list(const vector<T> &v, const string &name){
    size_t n = v.size();
    cout << name << " = [";
    for(size_t i = 0; i < n; ++i){
        if(i > 0)
            cout << ", ";
        cout << v[i];
    }
    cout << "]" << endl;
}

int main(){
    srand(time(0));
    rand();

    double eps = 0.03;

    unsigned int n = 10;
    vector<double> mcm, qm, kv, mc_err, q_err;

    double M = 2.0;

    using namespace chrono;

    high_resolution_clock::time_point t1, t2;

    cout.precision(17);
    //                 1        2         3     4          5          6      7           8           9      10
    vector<double> ref{1.0/3.0, 20.0/3.0, 27.0, 208.0/3.0, 425.0/3.0, 252.0, 1225.0/3.0, 1856.0 / 3, 891.0, 3700.0 / 3.0};

    for(unsigned int k = 1; k <= n; ++k){
        kv.push_back(k);
        vector<double> a(k), b(k);
        for(unsigned int i = 0; i < k; ++i){
            a[i] = 2 * i;
            b[i] = 2 * i + 1;
        }

        t1 = high_resolution_clock::now();
        double mc_res = monte_carlo_prec(f, a, b, eps);

        t2 = high_resolution_clock::now();
        mc_err.push_back(abs(ref[k-1] - mc_res));
        cout << "mc_res: " << mc_res << endl;
        mcm.push_back(duration_cast<microseconds>(t2 - t1).count());

        t1 = high_resolution_clock::now();
        Quadrature<decltype(f)> q(f, a, b, eps, M);
        double q_res = q.integral();

        t2 = high_resolution_clock::now();
        q_err.push_back(abs(ref[k-1] - q_res));
        cout << "q_res: " << q_res << endl;
        qm.push_back(duration_cast<microseconds>(t2 - t1).count());
        cout << endl; 
    }

    print_vec_as_list(kv, "kv");
    print_vec_as_list(mcm, "mcm");
    print_vec_as_list(qm, "qm");

    print_vec_as_list(mc_err, "mc_err");
    print_vec_as_list(q_err, "q_err");
}