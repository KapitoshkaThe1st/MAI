#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <cmath>

#include <iostream>

#include "eps.h"

enum class IntegrationMethod{
    Rectangle,
    Trapezoid,
    Simpson
};

template <typename T, class F>
T integral_rectangle(const F &f, T a, T b, T h = T(0.00001)) {
    size_t c = std::fabs(b - a) / h;

    T sum = T(0.0);
    T t = a + 0.5 * h;
    for (size_t i = 0; i < c; ++i) {
        sum += f(t + i * h);
    }

    return h * sum;
}

template <typename T, class F>
T integral_trapezoid(const F &f, T a, T b, T h = T(0.00001)) {
    size_t c = std::fabs(b - a) / h;

    T sum = T(0.0);
    for (size_t i = 0; i < c; ++i) {
        T t = i * h;
        sum += f(a + t) + f(a + t + h);
    }

    return h * T(0.5) * sum;
}

template <typename T, class F>
T integral_simpson(const F &f, T a, T b, T h = T(0.00001)) {
    size_t c = std::fabs(b - a) / h;

    T sum = T(0.0);
    for (size_t i = 0; i < c; ++i) {
        T t = i * h;
        sum += f(a + t) + T(4.0) * f(a + t + h * T(0.5)) + f(a + t + h);
    }

    return h * sum / T(6.0);
}

template<typename T>
T error_estimation(T s1, T s2, IntegrationMethod m){
    if(m == IntegrationMethod::Rectangle || m == IntegrationMethod::Trapezoid){
        return std::fabs(s1 - s2) / T(3.0);
    }
    else if(m == IntegrationMethod::Simpson){
        return std::fabs(s1 - s2) / T(15.0);
    }
    else{
        return nan("");
    }
}

template <typename T, class F>
T integral(const F &f, T a, T b, T eps = machine_eps<T>()){
    T h = std::sqrt(std::sqrt(eps)) * (b-a); // в какой-то книжке советовали начальный шаг брать таким
    T s = integral_simpson(f, a, b, h), ps;
    do{
        h *= 0.5;
        ps = s;
        s = integral_simpson(f, a, b, h);

        std::cout << "s: " <<  s << std::endl;
    }while(!approx_equal(error_estimation(s, ps, IntegrationMethod::Simpson), T(0.0), eps));

    return s;
}

#endif
