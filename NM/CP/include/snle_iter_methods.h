#ifndef SNLE_ITER_METHOD_H
#define SNLE_ITER_METHOD_H

#include <cmath>

#include "eps.h"
#include "math_utils.h"
#include "matrix.h"
#include "slae_iter_methods.h"
#include "deriv.h"

class SNLE{
private:
    static long iterations_required;

    template <typename T>
    static T approx_equal_vector(const std::vector<T> &a, const std::vector<T> &b, T max_absolute_diff = machine_eps<T>()) {
        std::vector<T> d = a-b;
        return approx_equal(std::sqrt(dot(d,d)), T(0.0), max_absolute_diff);
    }

public:
    template<typename T, class F>
    static std::vector<T> fixed_point_iter(const std::vector<F> &f, const std::vector<T> &init, T q, T prec){
        size_t n = f.size();

        std::vector<T> x = init, px;
        iterations_required = 0;

        T coef = (T(1.0) - q)/q;

        do{
            px = x;
            for(size_t i = 0; i < n; ++i)
                x[i] = f[i](px);
                // x[i] = f[i](x); // Не совсем по классике, в вычислениях учавствует каждый раз вектор с частично обновленными координатами. Этим похоже на метод зейделя (и действиетельно сходится быстрее)

            ++iterations_required;  
        }
        while(!approx_equal_vector(x, px, prec * coef));

        return x;
    }

    template<typename T, class F>
    static std::vector<T> newton_method(const std::vector<F> &f, const std::vector<T> &init, T prec){
        size_t n = f.size();

        Matrix<T> m(n);
        std::vector<T> b(n);
        std::vector<T> x = init, px;

        iterations_required = 0;
        do{
            jacobi_matrix(f, x, prec, m);

            for(size_t i = 0; i < n; ++i)
                b[i] = -f[i](x);

            px = x;
            x = x + SLAE::seidel_method(m, b, prec, true);

            ++iterations_required;
        }
        while(!approx_equal_vector(x, px, prec));

        return x;
    }

    static long iters_required(){
        return iterations_required;
    }
};

long SNLE::iterations_required = 0;

#endif
