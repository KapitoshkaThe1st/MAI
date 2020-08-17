#ifndef MATH_UTILS_H
#define MATH_UTILS_H
 
#include <complex>

template<typename T>
int sgn(T x){
    return (x > T(0)) - (x < T(0));
}

// solve quadratic equation
template<typename T>
std::complex<T> solve_qe(T a, T b, T c){
    T d = b*b - T(4.0) * a * c;
    T denum = a * T(2.0);
    T x = -b / denum;
    T y = std::sqrt(d > 0 ? d : -d) / denum;

    if(d < 0)
        return std::complex<T>(x, y);
    return std::complex<T>(x + y, T(0.0));
}

#endif
