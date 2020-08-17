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

    double eps_h = 0.01;
    int c = 1.0 / eps_h;
    vector<double> eps(c+1);
    for(int i = 0; i < c; ++i){
        eps[i] = 1.0 - eps_h * i;
        cout << eps[i] << ' ';
    }
    cout << endl;

    unsigned int n = 10;

    vector<vector<double>> mcm, qm;

    double M = 2.0;

    using namespace chrono;

    high_resolution_clock::time_point t1, t2;

    cout.precision(17);
    //                 1        2         3     4          5          6      7           8           9      10
    vector<double> ref{1.0/3.0, 20.0/3.0, 27.0, 208.0/3.0, 425.0/3.0, 252.0, 1225.0/3.0, 1856.0 / 3, 891.0, 3700.0 / 3.0};


    int avg_c = 3;
    for(unsigned int i = 1; i <= n; ++i){
        cerr << "dim: " << i << endl;
        mcm.push_back(vector<double>());
        qm.push_back(vector<double>());

        vector<double> a(i), b(i);
        for(unsigned int j = 0; j < i; ++j){
            a[j] = 2 * j;
            b[j] = 2 * j + 1;
        }

        unsigned long long mc_time = 0;
        unsigned long long q_time = 0;

        for(unsigned int k = 0; k < eps.size(); ++k){
            for(int j = 0; j < avg_c; ++j){
                t1 = high_resolution_clock::now();
                double mc_res = monte_carlo_prec(f, a, b, eps[k]);

                t2 = high_resolution_clock::now();
                mc_time += duration_cast<microseconds>(t2 - t1).count();
            }
            mcm[i-1].push_back((double)mc_time / avg_c);
            for(int j = 0; j < avg_c; ++j){
                t1 = high_resolution_clock::now();
                Quadrature<decltype(f)> q(f, a, b, eps[k], M);
                double q_res = q.integral();

                t2 = high_resolution_clock::now();
                q_time += duration_cast<microseconds>(t2 - t1).count();
            }
            qm[i-1].push_back((double)q_time / avg_c);
        }
    }

    print_vec_as_list(eps, "eps");
    cout << "mcm = [" << endl;
    for(size_t i = 0; i < mcm.size(); ++i){
        if(i > 0) 
            cout << "," << endl;
        cout << "[";
        for(size_t j = 0; j < mcm[i].size(); ++j){
            if(j > 0)
                cout << ", ";
            cout << mcm[i][j];
        }
        cout << "]";
    }
    cout << "]" << endl;
    
    cout << "qm = [" << endl;
    for(size_t i = 0; i < qm.size(); ++i){
        if(i > 0) 
            cout << "," << endl;
        cout << "[";
        for(size_t j = 0; j < qm[i].size(); ++j){
            if(j > 0)
                cout << ", ";
            cout << qm[i][j];
        }
        cout << "]";
    }
    cout << "]" << endl;
}