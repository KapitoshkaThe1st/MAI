#ifndef SLAE_ITER_METHODS
#define SLAE_ITER_METHODS

#include <vector>
#include <cmath>

#include "matrix.h"
#include "eps.h"

template<typename T>
class Matrix;

class SLAE{

private:
    template<typename T>
    static T vec_length(const std::vector<T> &v){
        T s = 0;
        for(T i : v)
            s += i*i;
        return sqrt(s);
    }

    template<typename T>
    static void prepare(Matrix<T> &m, std::vector<T> &b){
        size_t n = m.size();
        for(size_t i = 0; i < n; ++i){
            for(size_t j = 0; j < n; ++j){
                if(i != j)
                    m[i][j] = -m[i][j] / m[i][i];
            }
            b[i] = b[i] / m[i][i];
            m[i][i] = T(0.0);
        }
    }

    template<typename T>
    static void iterative_helper(Matrix<T> &m, std::vector<T> &b, T prec, std::vector<T> &x){
        
        prepare(m, b);

        T norm = m.norm_2();
        T coef = norm / (T(1.0) - norm);

        // std::cout << "norm: " << norm << std::endl;

        x = b;
        std::vector<T> prevX;

        iterations_required = 0;

        while(prevX.empty() || !approx_equal<T>(coef*vec_length(x - prevX), T(0.0), prec)){
        // while(prevX.empty() || !approx_equal<T>(vec_length(x - prevX), T(0.0), prec)){
            prevX = x;
            x = m * x + b;

            iterations_required++;
        }
    }

    template<typename T>
    static void seidel_helper(Matrix<T> &m, std::vector<T> &b, T prec, std::vector<T> &x){

        size_t n = m.size();

        prepare(m, b);

        T norm = m.norm_2();
        T coef = norm / (T(1.0) - norm);

        // std::cout << "norm: " << norm << std::endl;

        x = b;
        std::vector<T> prevX;

        iterations_required = 0;

        while(prevX.empty() || !approx_equal<T>(coef*vec_length(x - prevX), T(0.0), prec)){
        // while(prevX.empty() || !approx_equal<T>(vec_length(x - prevX), T(0.0), prec)){
            prevX = x;
            for(size_t i = 0; i < n; ++i){
                T sum = T(0.0);
                for(size_t j = 0; j < n; ++j){                
                    if(j < i)
                        sum += x[j] * m.data[i][j];
                    else
                        sum += prevX[j] * m.data[i][j];

                }
                x[i] = sum + b[i];
            }

            iterations_required++;
        }
    }

    static long iterations_required;

public:
    static long iters_required(){
        return iterations_required;
    }

    template<typename T>
    static std::vector<T> iterative_method(Matrix<T> &a, std::vector<T> &b, T precision=machine_eps<T>(), bool permute_inplace=false){
        size_t n = a.size();
        assert(n == b.size());

        std::vector<T> x;

        if(permute_inplace){
            iterative_helper(a, b, precision, x);
        }
        else{
            Matrix<T> m = a;
            std::vector<T> v = b;
            iterative_helper(m, v, precision, x);
        }

        return x;
    }

    template<typename T>
    static std::vector<T> seidel_method(Matrix<T> &a, std::vector<T> &b, T precision=machine_eps<T>(), bool permute_inplace=false){
        size_t n = a.size();
        assert(n == b.size());

        std::vector<T> x;

        if(permute_inplace){
            seidel_helper(a, b, precision, x);
        }
        else{
            Matrix<T> m = a;
            std::vector<T> v = b;
            seidel_helper(m, v, precision, x);
        }

        return x;
    }

};

long SLAE::iterations_required = 0;

#endif
