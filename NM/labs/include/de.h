#ifndef DE_H
#define DE_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

#include <iostream>

#include "eps.h"
#include "tdma.h"

template<typename T>
T operator_helper(const std::vector<std::vector<T>> &grid, size_t k, T x){
    if (x <= grid[0][0])
        return grid[0][k];

    if (x >= grid[grid.size() - 1][0])
        return grid[grid.size() - 1][k];

    auto it = std::lower_bound(grid.begin(), grid.end(), x, [](const auto &a, double val) -> bool { return a[0] < val; });
    size_t i = it - grid.begin();

    return grid[i - 1][k] + (grid[i][k] - grid[i - 1][k]) / (grid[i][0] - grid[i - 1][0]) * (x - grid[i - 1][0]);
}

// EULER-METHOD SODE SOLVER
template<typename T, class F>
class SODE_Euler{
private:
    std::vector<std::vector<T>> grid;
public:
    SODE_Euler(const std::vector<F> &f, const std::vector<T> &p, T a, T b, T h){
        size_t n = f.size();
        assert(n == p.size());
        
        size_t c = (b-a) / h + 1;

        grid = std::vector<std::vector<T>>(c);
        for(auto &v : grid)
            v = std::vector<T>(n); 
    
        grid[0] = p;

        for(size_t i = 1; i < c; ++i){
            for(size_t j = 0; j < n; ++j){
                grid[i][j] = grid[i-1][j] + h*f[j](grid[i-1]);
            }
        }
    }

    T operator()(size_t k, T x) {
        return operator_helper(grid, k, x);
    }
};

template <typename T, class F>
void runge_kutta_iter(size_t i, const std::vector<F> &f, std::vector<std::vector<T>> &grid, T h) {
    size_t n = f.size();

    std::vector<T> K1(n);
    std::vector<T> K2(n);
    std::vector<T> K3(n);
    std::vector<T> K4(n);

    std::vector<T> x = grid[i - 1];

    for (size_t k = 0; k < n; ++k)
        K1[k] = h * f[k](x);

    for (size_t k = 0; k < n; ++k)
        x[k] += 0.5 * K1[k];

    for (size_t k = 0; k < n; ++k)
        K2[k] = h * f[k](x);

    x = grid[i - 1];
    for (size_t k = 0; k < n; ++k)
        x[k] += 0.5 * K2[k];

    for (size_t k = 0; k < n; ++k)
        K3[k] = h * f[k](x);

    x = grid[i - 1];
    for (size_t k = 0; k < n; ++k)
        x[k] += K3[k];

    for (size_t k = 0; k < n; ++k)
        K4[k] = h * f[k](x);

    for (size_t j = 0; j < n; ++j) {
        grid[i][j] = grid[i - 1][j] + (K1[j] + 2.0 * (K2[j] + K3[j]) + K4[j]) / 6.0;
    }
}

// RUNGE-KUTTA-METHOD SODE SOLVER
template <typename T, class F>
class SODE_RungeKutta{
private:
    std::vector<std::vector<T>> grid;

public:
    SODE_RungeKutta(const std::vector<F> &f, const std::vector<T> &p, T a, T b, T h) {
        size_t n = f.size();
        assert(n == p.size());

        size_t c = (b - a) / h + 1;

        grid = std::vector<std::vector<T>>(c);
        for (auto &v : grid)
            v = std::vector<T>(n);

        grid[0] = p;

        for (size_t i = 1; i < c; ++i) {
            runge_kutta_iter(i, f, grid, h);
        }
    }

    T operator()(size_t k, T x) {
        return operator_helper(grid, k, x);
    }
};
// ADAMS-METHOD SODE SOLVER
template <typename T, class F>
class SODE_Adams {
private:
    std::vector<std::vector<T>> grid;

public:
    SODE_Adams(const std::vector<F> &f, const std::vector<T> &p, T a, T b, T h) {
        size_t n = f.size();
        assert(n == p.size());

        size_t c = (b - a) / h + 1;

        grid = std::vector<std::vector<T>>(c);
        for (auto &v : grid)
            v = std::vector<T>(n);

        grid[0] = p;

        for (size_t i = 1; i < 4; ++i) {
            runge_kutta_iter(i, f, grid, h);
        }

        for(size_t i = 4; i < c; ++i){
            for(size_t j = 0; j < n; ++j){
                grid[i][j] = grid[i - 1][j] + h * (55.0 * f[j](grid[i - 1]) - 59.0 * f[j](grid[i - 2]) + 37.0 * f[j](grid[i - 3]) - 9.0 * f[j](grid[i - 4])) / 24.0;
            }
        }

    }

    T operator()(size_t k, T x) {
        return operator_helper(grid, k, x);
    }
};

enum class SODEMethod{
    Euler, RungeKutta, Adams
};

template <typename T>
T error_estimation(T s1, T s2, SODEMethod m) {
    if (m == SODEMethod::Euler) {
        return std::fabs(s1 - s2) / T(3.0);
    } else if (m == SODEMethod::RungeKutta || m == SODEMethod::Adams) {
        return std::fabs(s1 - s2) / T(15.0);
    } else {
        return nan("");
    }
}

template <typename T, class F>
class BVP_ShootingMethod{
private:
    SODE_RungeKutta<T, F> rk;

    static SODE_RungeKutta<T, F> ctor(const std::vector<F> &f, T a0, T b0, T c0, T a1, T b1, T c1, T a, T b, T h, T eps) {
        T eta0 = 10.0;
        T xi0 = (c0 - a0*eta0)/b0;

        T eta1 = -10.0;
        T xi1 = (c0 - a0*eta1)/b0;

        std::vector<T> p1{a, xi0, eta0};
        std::vector<T> p2{a, xi1, eta1};

        SODE_RungeKutta<T, F> rk1(f, p1, a, b, h);

        int k = 0;
        while (true) {
            SODE_RungeKutta<T, F> rk2(f, p2, a, b, h);

            T t1 = rk1(2, b);
            T t2 = rk2(2, b);

            T q1 = rk1(1, b);
            T q2 = rk2(1, b);

            T w1 = a1 * t1 + b1 * q1 - c1;
            T w2 = a1 * t2 + b1 * q2 - c1;

            if (approx_equal(w2, w1, eps)) {
                rk1 = std::move(rk2);
                break;
            }

            T d = (p2[2] - p1[2]) / (w2 - w1) * w2;

            p1[2] = p2[2];
            p2[2] -= d;

            p1[1] = p2[1];
            p2[1] = (c0 - a0*p2[2])/b0;

            rk1 = std::move(rk2);
            ++k;
        }

        return rk1;
    }

public:
    BVP_ShootingMethod(const std::vector<F> &f, T a0, T b0, T c0, T a1, T b1, T c1, T a, T b, T h, T eps)
        : rk(ctor(f, a0, b0, c0, a1, b1, c1, a, b, h, eps)) { }


    T operator()(T x) {
        return rk(2, x);
    }
};


template <typename T, class F>
class BVP_FiniteDifference {
private:
    std::vector<T> x_;
    std::vector<T> y_;

public:
    BVP_FiniteDifference(const F &t, const F &p, const F &q, const F &f, T a0, T b0, T c0, T a1, T b1, T c1, T lb, T rb, T h){
        size_t n = (rb - lb) / h;
        
        std::vector<T> k(n), l(n+1), m(n), g(n+1);

        x_ = std::vector<T>(n+1);
        for(size_t i = 0; i < n+1; ++i)
            x_[i] = lb + i*h;

        T t0 = t(x_[0]);
        T k0 = (t0 / (h*h) - p(x_[0]) / (2*h));
        T l0 = q(x_[0]) - 2.0*t0/(h*h);
        T m0 = (t0 / (h*h) + p(x_[0]) / (2*h));

        l[0] = k0*a0 + l0*b0/(2*h);
        m[0] = b0/(2*h) * (k0 + m0);
        g[0] = c0 * k0 + b0/(2*h) * f(x_[0]);

        for(size_t i = 1; i < n; ++i){
            T ti = t(x_[i]);
            k[i-1] = (ti / (h*h) - p(x_[i]) / (2*h));
            l[i] = q(x_[i]) - 2.0*ti/(h*h);
            m[i] = (ti / (h*h) + p(x_[i]) / (2*h));
            g[i] = f(x_[i]);
        }

        T tn = t(x_[n]);
        T kn = (tn / (h*h) - p(x_[n]) / (2*h));
        T ln = q(x_[n]) - 2.0*tn/(h*h);
        T mn = (tn / (h*h) + p(x_[n]) / (2*h));

        k[n-1] = b1/(2*h) * (mn + kn);
        l[n] = ln*b1/(2*h) - a1*mn;
        g[n] = b1/(2*h) * f(x_[n]) - c1*mn;

        y_ = tdma(k, l, m, g, n+1);
    }

    T operator()(T x){
        if(x <= x_[0])
            return y_[0];
        if(x >= x_[x_.size()-1])
            return y_[x_.size()-1];

        auto it = lower_bound(x_.begin(), x_.end(), x);
        size_t i = it - x_.begin();

        return y_[i-1] + (y_[i] - y_[i-1]) / (x_[i] - x_[i-1]) * (x - x_[i-1]);
    }
};

#endif