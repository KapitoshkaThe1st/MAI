#ifndef NLE_ITER_METHODS_H
#define NLE_ITER_METHODS_H

#include "eps.h"
#include "math_utils.h"

class NLE{
private:
    static long iterations_required;
public:
    template<typename T, class F>
    static T fixed_point_iter(F f, T init, T q, T prec){
        T x = init, px;

        T coef = (T(1.0)-q)/q;

        iterations_required = 0;
        do{
            px = x;
            x = f(x);
            ++iterations_required;
        }
        while(!approx_equal(x, px, prec * coef));

        return x;
    }

    template<typename T, class F>
    static T newton_method(F f, T init, T prec){
        T x = init, px;
        
        iterations_required = 0;
        do{
            T fx = f(x);
            px = x;
            x = x - fx / (f(x+prec) - fx) * prec;
            ++iterations_required;
        }
        while(!approx_equal(x, px, prec));

        return x;
    }

    template<typename T, class F>
    static T dichotomy_method(F f, T a, T b, T prec){
        iterations_required = 0;
        while(!approx_equal(a, b, T(2.0)*prec)){
            T fa = f(a), m = (a+b)/T(2.0), fm = f(m);
            if(sgn(fa) == sgn(fm))
                a = m;
            else
                b = m;
            ++iterations_required;
        }

        return (a+b)/T(2.0);
    }

    static long iters_required(){
        return iterations_required;
    }
};

long NLE::iterations_required = 0; 

#endif
