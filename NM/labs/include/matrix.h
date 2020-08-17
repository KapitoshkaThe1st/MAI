#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <limits>

#include "utils.h"
#include "eps.h"
#include "slae_iter_methods.h"
#include "eigenv.h"

template<typename T> class LU;
class SLAE;
class EigenV;

template<typename T>
class Matrix{

    using vec = std::vector<T*>;

    class Row{
    private: 
        T *ptr;
        size_t n;
    public:
        Row(vec &data, size_t i){
            n = data.size();
            ptr = data[i]; 
        }

        T& operator[](size_t i){
            assert(i >= 0 && i < n);
            return ptr[i]; 
        }
    };

private:
    vec data;
    T *orig_ptr;

    void ctor_helper(size_t n){
        data = vec(n);
        orig_ptr = new T[n*n];

        for(size_t i = 0; i < n; ++i){
            data[i] = orig_ptr + i * n;
        }
    }

    void copy_helper(const Matrix &other){
        size_t n = other.data.size();

        data = vec(n);
        orig_ptr = new T[n*n];
        
        T *src = other.orig_ptr;
        long offset = orig_ptr - src;

        memcpy(orig_ptr, src, n*n * sizeof(T));

        for(size_t i = 0; i < n; ++i){
            data[i] = other.data[i] + offset; 
        }
    }

public:
    Matrix(size_t n){
        ctor_helper(n);
    }

    Matrix(size_t n, T val){
        ctor_helper(n);
        T *lim = orig_ptr + n*n;
        for(T *ptr = orig_ptr; ptr < lim; ++ptr)
            *ptr = val;
    }

    Matrix(std::istream &istr) {
        size_t n;
        istr >> n;

        ctor_helper(n);

        T *lim = orig_ptr + n * n;

        for(T *ptr = orig_ptr; ptr < lim; ++ptr){
            istr >> *ptr;
        }
    }

    ~Matrix(){
        delete[] orig_ptr;
    }

    Matrix(const Matrix &other){
        // cout << "copy ctor" << endl;
        copy_helper(other);
    }

    Matrix(Matrix &&other){
        // cout << "move ctor" << endl;

        data = std::move(other.data);
        orig_ptr = other.orig_ptr;
        other.orig_ptr = nullptr;
    }

    size_t size() const {
        return data.size();
    }

    T norm_1() const {
        size_t n = size();

        T maxVal = T(0.0);
        for(size_t j = 0; j < n; ++j){
            T sum = T(0.0);
            for(size_t i = 0; i < n; ++i){
                sum += std::fabs(data[i][j]);
            }
            if(sum > maxVal)
                maxVal = sum;
        }

        return maxVal;
    }

    T norm_2() const {
        size_t n = size();

        T sum = T(0.0);
        for(size_t j = 0; j < n; ++j){
            for(size_t i = 0; i < n; ++i){
                T el = data[i][j];
                sum += el*el;
            }
        }

        return std::sqrt(sum);
    }

    T norm_c() const {
        size_t n = size();

        T maxVal = T(0.0);

        for(size_t i = 0; i < n; ++i){
            T sum = T(0.0);
            for(size_t j = 0; j < n; ++j){
                sum += std::fabs(data[i][j]);
            }
            if(sum > maxVal)
                maxVal = sum;
        }

        return maxVal;
    }

    Matrix& operator=(const Matrix &other){
        // cout << "copy assignment" << endl;
        
        if(orig_ptr != nullptr)
            delete[] orig_ptr;

        copy_helper(other);

        return *this;
    }

    Matrix& operator=(Matrix &&other){
        // cout << "move assignment" << endl;

        if(orig_ptr != nullptr)
            delete[] orig_ptr;

        data = std::move(other.data);
        orig_ptr = other.orig_ptr;
        other.orig_ptr = nullptr;

        return *this;
    }

    Row operator[](size_t i){
        size_t n = data.size();
        assert(i >= 0 && i < n);
        return Row(data, i); 
    }

    friend std::ostream &operator<<(std::ostream &ostr, const Matrix<T> &m) {
        size_t n = m.data.size();

        ostr << n << std::endl;

        for(T *ptr : m.data){
            if(ptr != m.data[0])
                ostr << std::endl;
            for(T *i = ptr; i < ptr + n; ++i){
                ostr << *i << ' ';
            }
        }

        return ostr;
    }

    template<typename U>
    friend Matrix<U> operator*(const Matrix<U>&, const Matrix<U>&);
    template<typename U>
    friend std::vector<U> operator*(const Matrix<U>&, const std::vector<U>&);
    // template<typename U>
    // friend vector<U> seidel_helper(const Matrix<U>&, const vector<U>&, const vector<U>&);

    template<typename U>
    friend Matrix<U> operator*(const U, const Matrix<U>&);
    template<typename U>
    friend Matrix<U> operator-(const Matrix<U>&, const Matrix<U>&);
    template<typename U>
    friend Matrix<U> operator+(const Matrix<U>&, const Matrix<U>&);
    template<typename U>

    friend Matrix<U> operator*(const std::vector<U>&, const std::vector<U>&);



    friend class LU<T>;
    friend class SLAE;
    friend class EigenV;
};

template<typename T>
Matrix<T> operator*(const Matrix<T> &m1, const Matrix<T> &m2){
    size_t n = m2.data.size();
    assert(n == m1.data.size());
    
    Matrix<T> nm(n);
    for(size_t  i = 0; i < n; ++i)
        for(size_t  j = 0; j < n; ++j){
            T sum = 0;
            for(size_t  k = 0; k < n; ++k)
                sum += m1.data[i][k] * m2.data[k][j];
            nm.data[i][j] = sum;
        }                     
    return nm;
}

template<typename T>
Matrix<T> operator*(const T a, const Matrix<T> &m){
    size_t n = m.data.size();

    Matrix<T> nm(n);
    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < n; ++j)
            nm.data[i][j] = a * m.data[i][j];     

    return nm;
}

template<typename T>
Matrix<T> operator+(const Matrix<T> &m1, const Matrix<T> &m2){
    size_t n = m2.data.size();
    assert(n == m1.data.size());

    Matrix<T> nm(n);
    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < n; ++j)
            nm.data[i][j] = m1.data[i][j] + m2.data[i][j];                   
    
    return nm;
}

template<typename T>
Matrix<T> operator-(const Matrix<T> &m1, const Matrix<T> &m2){
    size_t n = m2.data.size();
    assert(n == m1.data.size());

    Matrix<T> nm(n);
    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < n; ++j)
            nm.data[i][j] = m1.data[i][j] - m2.data[i][j];                   
    
    return nm;
}

template <typename T>
std::vector<T> operator*(const Matrix<T> &m, const std::vector<T> &v) {
    size_t n1 = m.data.size();
    size_t n = v.size();
    
    assert(n1 = n);

    std::vector<T> res(n);
    for(size_t i = 0; i < n; ++i){
        T sum = 0;
        for(size_t j = 0; j < n; ++j){
            sum += m.data[i][j] * v[j];
        }
        res[i] = sum;
    }
    return res;
}

template <typename T>
std::ostream &operator<<(std::ostream &ostr, const std::vector<T> &v) {
    for(auto &el : v)
        ostr << el << ' ';

    return ostr;
}

template<typename T>
class LU{
private:
    size_t permut_count;
    Matrix<T> mat;
    std::vector<int> permut;

    std::vector<int> LU_decompose(Matrix<T> &m) {
        size_t n = m.data.size();

        std::vector<int> permut(n, -1);
        size_t p = 0;

        for(size_t i = 0; i < n-1; ++i){
            size_t maxInd = i;
            T maxVal = std::fabs(m.data[i][i]);

            for(size_t j = i+1; j < n; ++j){
                T val = std::fabs(m.data[j][i]);
                if(val > maxVal){
                    maxInd = j;
                    maxVal = val;
                }
            }

            if(approx_equal(maxVal, T(0.0)))
                throw RUNTIME_ERROR("Division by zero: ");
            

            if(maxInd != i){
                std::swap(m.data[i], m.data[maxInd]);
                permut[i] = maxInd;
                p++;
            }

            for(size_t j = i+1; j < n; ++j){
                T coef = m.data[j][i] / m.data[i][i];

                m.data[j][i] = coef;
                for(size_t k = i+1; k < n; ++k){
                    m.data[j][k] = m.data[j][k] - coef * m.data[i][k];
                }
            }
        }

        return permut;
    }

    void ctor_helper(){
        permut = LU_decompose(mat);
        permut_count = 0;
        for(int i : permut)
            if(i != -1)
                permut_count++;
    }

public:
    LU(Matrix<T> &m) : mat(m) {
        ctor_helper();
    }

    LU(Matrix<T> &&m) : mat(std::move(m)) {
        ctor_helper();
    }

    size_t size() const {
        return mat.data.size();
    }

    void slae_solve_helper(std::vector<T> &v, std::vector<T> &x) {
        size_t n = v.size();

        // permutatins exact like when lu-decomposing
        for(size_t i = 0; i < n; ++i){
            if(permut[i] != -1){
                std::swap(v[i], v[permut[i]]);
            }
        }

        // Ly = b
        for(size_t i = 0; i < n; ++i){
            T sum = 0;
            for(size_t j = 0; j < i; ++j){
                sum += mat.data[i][j] * x[j];
            }
            x[i] = v[i] - sum;
        }

        // Ux = y
        for(size_t i = n-1; i != std::numeric_limits<size_t>::max(); --i){
            T sum = 0;
            for(size_t j = i+1; j < n; ++j){
                sum += mat.data[i][j] * x[j];
            }
            x[i] = (x[i] - sum) / mat.data[i][i];
        }
    }

    std::vector<T> slae_solve(std::vector<T> &v, bool permute_inplace = false) {
        size_t n = mat.size();
        std::vector<T> x(n);
        if(permute_inplace){
            slae_solve_helper(v, x);
        }   
        else{
            std::vector<T> temp = v;
            slae_solve_helper(temp, x);
        }

        return x;
    }

    T det() const {
        size_t n = mat.size();

        T res = T(1.0);
        for(size_t i = 0; i < n; ++i)
            res *= mat.data[i][i];

        return permut_count & 1 ? -res : res;
    }

    Matrix<T> inverse(){
        size_t n = mat.data.size();
        Matrix<T> nm(n);

        // if(approx_equal(det(), 0.0, eps))
        //     throw RUNTIME_ERROR("Determinant is zero: ");

        std::vector<T> v(n, T(0.0));
        for(size_t i = 0; i < n; ++i){
            v[i] = T(1.0);
            std::vector<T> x = slae_solve(v);

            v[i] = T(0.0);
            for(size_t j = 0; j < n; ++j){
                nm.data[j][i] = x[j];
            }
        }

        return nm;
    }
};

template<typename T>
T dot(const std::vector<T> &a, const std::vector<T> &b){
    size_t n = b.size();
    assert(n == a.size());

    T res = T(0.0);
    for(size_t i = 0; i < n; ++i)
        res += a[i] * b[i];
    
    return res;
}

template <typename T>
std::vector<T> operator-(const std::vector<T> &a, const std::vector<T> &b) {
    size_t n = a.size();
    assert(n == b.size());

    std::vector<T> res(n);

    for(size_t i = 0; i < n; ++i){
        res[i] = a[i] - b[i];
    }

    return res;
}

template <typename T>
std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b) {
    size_t n = a.size();
    assert(n == b.size());

    std::vector<T> res(n);

    for(size_t i = 0; i < n; ++i){
        res[i] = a[i] + b[i];
    }

    return res;
}

template <typename T>
std::vector<T> operator*(const T a, const std::vector<T> &b) {
    size_t n = b.size();
    assert(n == b.size());

    std::vector<T> res(n);

    for(size_t i = 0; i < n; ++i){
        res[i] = a * b[i];
    }

    return res;
}

template<typename T>
Matrix<T> operator*(const std::vector<T> &a, const std::vector<T> &b){
    size_t n = b.size();
    assert(n == b.size());

    Matrix<T> res(n);

    for(size_t i = 0; i < n; ++i)
        for(size_t j = 0; j < n; ++j)
            res.data[i][j] = a[i] * b[j];

    return res;
}

#endif
