#ifndef DERIV_H
#define DERIV_H

#include "matrix.h"

#include <vector>
#include <algorithm>
#include <limits>

#include <iostream>

template <typename T, typename F>
T simple_deriv(F &f, T x, T prec, size_t n = 1) {
    if(n == 0)
        return f(x);
    
    return (simple_deriv(f, x + prec, prec, n-1) - simple_deriv(f, x - prec, prec, n-1)) / (T(2.0)*prec);
}

template<typename T, typename F>
T simple_partial_deriv(const F &f, std::vector<T> x, T prec, size_t i, size_t n=1){
    if(n == 0)
        return f(x);
    
    x[i] += prec;
    T dplus = simple_partial_deriv(f, x, prec, i, n-1);
    x[i] -= T(2.0)*prec;
    T dminus = simple_partial_deriv(f, x, prec, i, n-1);

    return (dplus - dminus) / (T(2.0)*prec);
}

template<typename T, typename F>
void jacobi_matrix(const std::vector<F> &f, const std::vector<T> &x, T prec, Matrix<T> &m){
    size_t n = f.size();
    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < n; ++j)
            m[i][j] = simple_partial_deriv(f[i], x, prec, j, 1);
}

template<typename T>
T deriv(const std::vector<T> &x, const std::vector<T> &f, T p){
    size_t n = x.size();
    assert(n == f.size());

    auto it = std::lower_bound(x.begin(), x.end(), p);
    size_t i = it - x.begin();

    if(i-1 == std::numeric_limits<size_t>::max() || i >= n)
        return std::nan("");

    return (f[i] - f[i - 1]) / (x[i] - x[i - 1]);
}

template <typename T>
T deriv2(const std::vector<T> &x, const std::vector<T> &f, T p) {
    size_t n = x.size(); 
    assert(n == f.size());

    auto it = std::lower_bound(x.begin(), x.end(), p);
    size_t i = it - x.begin();

    if (i - 1 == std::numeric_limits<size_t>::max() || i + 1 >= n)
        return std::nan("");

    T l = (f[i+1] - f[i]) / (x[i+1] - x[i]);
    T r = (f[i] - f[i-1]) / (x[i] - x[i-1]);

    return T(2.0) * (l - r) / (x[i+1] - x[i-1]);
}

#endif
