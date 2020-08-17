#ifndef LSM_H
#define LSM_H

#include <vector>
#include <iostream>

#include "matrix.h"

template<typename T>
class LSM{
private:
    std::vector<T> coef;
public:

    LSM(const std::vector<T> x, const std::vector<T> f, size_t m){
        size_t n = x.size();
        assert(n == f.size());

        Matrix<T> M(m);
        std::vector<T> b(m);

        for(size_t i = 0; i < m; ++i){
            for(size_t j = 0; j < m; ++j){
                T sum = T(0.0);
                for(size_t k = 0; k < n; ++k){
                    sum += std::pow(x[k], i + j);
                }
                M[i][j] = sum;
            }
        }
        
        for(size_t i = 0; i < m; ++i){
            T sum = T(0.0);
            for(size_t k = 0; k < n; ++k){
                sum += f[k] * std::pow(x[k], i);
            }
            b[i] = sum;
        }

        LU lu(M);
        coef = lu.slae_solve(b, true);
    }

    T operator()(T x){
        size_t m = coef.size();
        T res = coef[m-1];
        for(size_t i = 1; i < m; ++i)
            res = res * x + coef[m-i-1];
        return res;
    }

    void print_coefs(){
        for(T e : coef)
            std::cout << e << ' ';
        std::cout << std::endl;
    }
};

#endif
