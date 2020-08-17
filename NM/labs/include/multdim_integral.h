#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <vector>
#include <cmath>
#include <cassert>
#include <random>
#include <cstdlib>

#include "deriv.h"

double random_double(double a, double b){
    static std::mt19937 gen(time(0));
    static const double norm_coef = 1.0 / gen.max();

    return a + (double)gen() * (b - a) * norm_coef;
}

void random_vector(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &v){
    size_t n = v.size();
    for(size_t i = 0; i < n; ++i){
        v[i] = random_double(a[i], b[i]);
    }
}

template<class F>
double monte_carlo_method(const F &f, const std::vector<double> &a, const std::vector<double> &b, unsigned long long k){
    size_t n = a.size();
    assert(n == b.size());
    std::vector<double> x(n);
    double s = 0.0;

    double v = 1.0;
    for(size_t i = 0; i < n; ++i){
        v *= b[i] - a[i];
    }

    for(unsigned long long i = 0; i < k; ++i){
        random_vector(a, b, x);
        s += f(x);
    }

    return s * v / k;
}

template<class F>
double monte_carlo_prec(const F &f, const std::vector<double> &a, const std::vector<double> &b, double eps){
    size_t n = a.size();
    double p = 1.0;
    double s = 0.0;

    std::vector<double> x(n);
    for(size_t i = 0; i < n; ++i)
        x[i] = (a[i] + b[i]) / 2.0;

    for(size_t i = 0; i < n; ++i){
        double t = b[i] - a[i];
        p *= t * t;
        double d = simple_partial_deriv(f, x, 0.001, i);

        s += d*d * t*t;
    }

    double D = p * s / 12.0;

    unsigned long long k = ceil(9.0 * D/(eps * eps));

    return monte_carlo_method(f, a, b, k);
}

template<class F>
class Quadrature{
private:
    const F &_f;
    std::vector<double> _a, _b, _h;
    double sum;
    double _M;
    double rh;

    std::vector<double> x;
    unsigned int k;
double aux(){
        double s = 0.0;
        double w = _b[k] - _a[k];
        unsigned int c = round(w / _h[k]);
        if(k < _a.size()-1){
            ++k;
            s += 0.5 * aux();
            for(unsigned int i = 1; i < c; ++i){
                x[k] = _a[k] + i * _h[k];
                ++k;
                s += aux();
            }
            x[k] = _b[k];
            ++k;
            s += 0.5 * aux();
        }
        else{
            s += 0.5 * _f(x);
            for(unsigned int i = 1; i < c; ++i){
                x[k] = _a[k] + i * _h[k];
                s += _f(x);
            }
            x[k] = _b[k];
            s += 0.5 * _f(x);
        }
        s *= _h[k];

        x[k] = _a[k];
        --k;

        return s;
    }

    double proper_h(double eps, double M){
        size_t n = _a.size();
        double V = 1.0;
        for(unsigned int i = 0; i < n; ++i){
            V *= _b[i] - _a[i];
        }

        return sqrt(12.0 * eps / (M * n * V));
    }

public:
    Quadrature(const F &f, std::vector<double> &a, std::vector<double> &b, double eps, double M)
    : _f(f), _a(a), _b(b), _M(M) {
        size_t n = a.size();
        rh = proper_h(eps, _M);
        _h = std::vector<double>(n);


        for(size_t i = 0; i < n; ++i){
            double w = _b[i] - _a[i];
            _h[i] = w / ceil(w / rh);
        }

        x = _a;
        k = 0;
        sum = aux();
    }

    double integral(){
        return sum;
    }

    double err(){
        size_t n = _a.size();
        double p = 1.0;
        for(size_t i = 0; i < n; ++i){
            p *= _b[i] - _a[i];
        }

        return _M * rh * rh * n * p / 12.0;
    }
};

#endif