#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include <vector>
#include <cassert>

#include "matrix.h"
#include "tdma.h"

#include <iostream>

using namespace std;

template<typename T>
class LagrangePolynomial{
private:
    size_t n;
    std::vector<T> f_;
    std::vector<T> w_;
    std::vector<T> x_;

    T prod(T x, size_t k){
        T p = T(1.0);
        for(size_t i = 0; i < n; ++i){
            if(k != i)
                p *= x - x_[i];
        }
        return p;
    }

public:
    LagrangePolynomial(const std::vector<T> x, const std::vector<T> f) {
        n = x.size();
        assert(n == f.size());

        f_ = f;
        x_ = x;
        w_ = std::vector<T>(n);
        for(size_t i = 0; i < n; ++i){
            T p = T(1.0);
            for(size_t j = 0; j < n; ++j){
                if(i != j)
                    p *= x_[i] - x_[j];
            }
            w_[i] = p;
        }
    }

    T operator()(T x){
        T s = T(0.0);
        for(size_t i = 0; i < n; ++i){
            // cout << "W_" << i << ": " << w_[i] << endl;
            s += f_[i] * prod(x, i) / w_[i];
        }

        return s;
    }
};

template<typename T>
class NewtonPolynomial {
private:
    size_t n;
    std::vector<T> x_;
    std::vector<std::vector<T>> f_;
public:
    NewtonPolynomial() : n(0) { }

    NewtonPolynomial(const std::vector<T> &x, const std::vector<T> &f){
        n = x.size();
        assert(n == f.size());
        x_ = x;
        f_ = std::vector<std::vector<T>>(n);
        for(size_t i = 0; i < n; ++i){
            f_[i] = std::vector<T>(n-i);
            f_[i][0] = f[i];
        }

        for(size_t j = 1; j < n; ++j){
            for(size_t i = 0; i < n - j; ++i){
                f_[i][j] = (f_[i][j-1] - f_[i+1][j-1]) / (x_[i] - x_[j+i]);
            }
        }
    }

    void add_point(T x, T f){
        ++n;
        x_.push_back(x);
        f_.push_back(std::vector<T>(1, f));
        for(size_t j = 1; j < n; ++j){
            size_t i = n - j - 1;
            f_[i].push_back(0);

            f_[i][j] = (f_[i][j-1] - f_[i+1][j-1]) / (x_[i] - x_[j+i]);
        }
    }

    T operator()(T x){
        T s = T(0.0);
        T p = T(1.0);
        for(size_t i = 0; i < n; ++i){
            s += p * f_[0][i];
            p *= x - x_[i];
        }
        return s;
    }
};

template <typename T>
class CubicSpline{
private:
    struct Segment{
        T a, b, c, d;
    };

    std::vector<T> x_;
    std::vector<Segment> seg_;

    static double h(const std::vector<T> &x, size_t i){
        return x[i] - x[i-1];
    }
public:
    CubicSpline(const std::vector<T> &x, const std::vector<T> &f){
        size_t n = x.size();
        assert(n == f.size());

        // cout << n << endl;

        n -= 2;

        std::vector<T> a(n-1), b(n), c(n-1), d(n); 

        b[0] = 2.0 * (h(x, 1) + h(x, 2));
        c[0] = h(x, 2);

        for(size_t i = 1; i < n-1; ++i){
            a[i-1] = h(x, i+1);
            b[i] = 2.0 * (h(x, i+1) + h(x, i+2));
            c[i] = h(x, i+2);
        }

        a[n-2] = h(x, n-2);
        b[n-1] = 2.0 * (h(x, n-2) + h(x, n-1));

        for(size_t i = 0; i < n; ++i){
            d[i] = 3.0 * ((f[i+2] - f[i+1])/h(x, i+2) - (f[i+1] - f[i]) / h(x, i+1));
        }

        // cout << "a: " << a << endl;
        // cout << "b: " << b << endl;
        // cout << "c: " << c << endl;
        // cout << "d: " << d << endl;

        std::vector<T> cc = tdma(a, b, c, d, n);
        cc.insert(cc.begin(), 0.0);
        // cout << "cc.size(): " << cc.size() << endl; 

        // cout << "cc: " << cc << endl;

        n += 2;
        seg_ = std::vector<Segment>(n-1);
        for(size_t i = 0; i < n-2; ++i){
            seg_[i].a = f[i];
            seg_[i].b = (f[i+1] - f[i]) / h(x, i+1) - h(x, i+1)*(cc[i+1] + 2.0 * cc[i]) / 3.0;
            seg_[i].c = cc[i];
            seg_[i].d = (cc[i+1] - cc[i]) / (3.0 * h(x, i+1));
        }


        seg_[n-2].a = f[n-2];
        seg_[n-2].b = (f[n-1] - f[n-2]) / h(x, n-1) - 2.0 * h(x, n-1) * cc[n-2] / 3.0;
        seg_[n-2].c = cc[n-2];
        seg_[n-2].d = -cc[n-2]/ (3.0 * h(x, n-1));
    
        // for(auto &s : seg_){
        //     cout << s.a << ' ' << s.b << ' ' << s.c << ' ' << s.d << endl;
        // }

        x_ = x;
    }

    T operator()(T x){
        auto it = lower_bound(x_.begin(), x_.end(), x);
        size_t b = it - x_.begin();
        size_t a = b-1;

        Segment &s = seg_[a];

        T h = (x - x_[a]);

        return s.a + s.b * h + s.c * h*h + s.d * h*h*h;
    }
};

#endif
