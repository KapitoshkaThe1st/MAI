#ifndef EPS_H
#define EPS_H

#include <limits>
#include <cmath>
#include <iostream>

// хорошо объясняется как правильно сравнивать числа с плавающей точкой и почему
// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/

template<typename T>
constexpr T machine_eps(){
    T d = T(1.0);
    while(T(1.0) + d / T(2.0) > T(1.0)){
        d /= T(2.0);
    }

    return d;
}

template<typename T>
bool approx_equal(T a, T b, T max_absolute_diff=machine_eps<T>(), T max_relative_diff=machine_eps<T>()){

    T diff = fabs(a - b);

    if(diff < max_absolute_diff)
        return true;

    a = fabs(a);
    b = fabs(b);
    T max = a > b ? a : b;

    if(diff < max_relative_diff * max)
        return true;

    return false;
}

template<typename T, typename U>
bool approx_equal_ULP(T a, T b, U, T max_absolute_diff=machine_eps<T>(), U max_ULP_diff=1){

    T diff = fabs(a - b);

    if(diff <= max_absolute_diff)
        return true;

    if((a < 0) != (b < 0))
        return false;

    // size of integer type (U) must be the same as float-point (T)
    U *a_ptr = (U*)&a;
    U *b_ptr = (U*)&b;

    if(abs(*a_ptr - *b_ptr) <= max_ULP_diff)
        return true;

    return false;
}

#endif