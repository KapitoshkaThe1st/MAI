#ifndef EIGEN_V
#define EIGEN_V

#include <vector>
#include <cmath>
#include <tuple>
#include <complex>

#include "matrix.h"
#include "eps.h"
#include "math_utils.h"


class EigenV{
private:
    template<typename T>
    static bool jacobi_criteria(const Matrix<T> &m, T prec){
        size_t n = m.size();

        T sum = T(0.0);
        for(size_t i = 0; i < n-1; ++i)
            for(size_t j = i+1; j < n; ++j){
                T val = m.data[i][j];
                sum += val * val;
            }

        sum = std::sqrt(T(2.0) * sum);
        if(approx_equal(sum, T(0.0), prec)){
            return false;
        }
        return true;
    }

    template<typename T>
    // Jacobi eigenvalue algorithm
    static void jacobi_helper(Matrix<T> &m, std::vector<T> &eval, std::vector<std::vector<T>> &evec, T prec){
        size_t n = m.size();
    
        Matrix<T> U(0);

        iterations_required = 0;

        while(jacobi_criteria(m, prec)){

            size_t mi, mj;
            T maxVal = std::numeric_limits<T>::min();
            for(size_t i = 0; i < n-1; ++i){
                for(size_t j = i+1; j < n; ++j){
                    T val = fabs(m.data[i][j]);
                    if(val > maxVal){
                        maxVal = val;
                        mi = i;
                        mj = j;
                    }
                }
            }

            Matrix<T> u(n, T(0.0));
            for(size_t i = 0; i < n; ++i){
                u.data[i][i] = T(1.0);
            }

            T denum = m.data[mi][mi] - m.data[mj][mj];
            
            T phi;
            if(approx_equal(denum, T(0.0)))
                phi = std::atan(T(1.0)); // pi/4
            else
                phi = T(0.5) * std::atan(T(2.0) * m.data[mi][mj] / denum);
            
            T c = std::cos(phi);
            T s = std::sin(phi);

            u.data[mi][mi] = u.data[mj][mj] = c;
            u.data[mi][mj] = -s;
            u.data[mj][mi] = s; 

            U = U.size() > 0 ? U * u : u;

            m = m * u;

            // inverse (transpose) rotation matrix
            std::swap(u.data[mi][mj], u.data[mj][mi]);

            m = u * m;

            iterations_required++;
        }

        eval = std::vector<T>(n);
        for(size_t i = 0; i < n; ++i)
            eval[i] = m.data[i][i];

        evec = std::vector<std::vector<T>>(n);
        for(size_t i = 0; i < n; ++i){
            evec[i] = std::vector<T>(n);
            for(size_t j = 0; j < n; ++j){
                evec[i][j] = U.data[j][i];
            }
        }
    }

    template<typename T>
    static Matrix<T> householder(const std::vector<T> &a, size_t k){
        size_t n = a.size();
        Matrix<T> E(n, T(0.0));
        for(size_t i = 0; i < n; ++i)
            E.data[i][i] = T(1.0);
    
        std::vector<T> v(n, T(0.0));
        for(size_t i = k; i < n; ++i)
            v[i] = a[i];

        // квадрат нормы нижнего подвектора
        T sum = T(0.0);
        for(size_t i = k; i < n; ++i){
            sum += a[i] * a[i];
        }

        v[k] += sgn(a[k]) * std::sqrt(sum); 

        T denum = dot(v, v);

        // ----------------------------
        // Matrix<T> M = v * v;
        Matrix<T> M(n);
        for(size_t i = 0; i < n; ++i)
            for(size_t j = i+1; j < n; ++j){
                T val = v[i] * v[j];
                M.data[i][j] = val;
                M.data[j][i] = val; 
            }
        for(size_t i = 0; i < n; ++i)
            M.data[i][i] = v[i] * v[i];
        // -----------------------------

        // return E - (T(2.0) / denum) * M;
        M = -(T(2.0) / denum) * M;
        for(size_t i = 0; i < n; ++i)
            M.data[i][i] += T(1.0);

        return M;
    }

    template<typename T>
    static Matrix<T> householder_mult_left(const Matrix<T> &h, const Matrix<T> &a, size_t m){
        size_t n = a.size();
        assert(n == h.size());

        Matrix<T> r(n, T(0.0));
        
        for(size_t i = 0; i < m; ++i){
            for(size_t j = 0; j < n; ++j){
                r.data[i][j] = a.data[i][j];
            }
        }
        for(size_t i = m; i < n; ++i){
            for(size_t j = 0; j < n; ++j){
                T sum = T(0.0);
                for(size_t k = m; k < n; ++k){
                    sum += h.data[i][k] * a.data[k][j];
                }
                r.data[i][j] = sum;
            }
        }

        return r;
    }

    template<typename T>
    static Matrix<T> householder_mult_right(const Matrix<T> &a, const Matrix<T> &h, size_t m){
        size_t n = a.size();
        assert(n == h.size());

        Matrix<T> r(n, T(0.0));
        
        for(size_t i = 0; i < n; ++i){
            for(size_t j = 0; j < m; ++j){
                r.data[i][j] = a.data[i][j];
            }
        }
        for(size_t i = 0; i < n; ++i){
            for(size_t j = m; j < n; ++j){
                T sum = T(0.0);
                for(size_t k = m; k < n; ++k){
                    sum += a.data[i][k] * h.data[k][j];
                }
                r.data[i][j] = sum;
            }
        }

        return r;
    }

    template<typename T>
    static void hessenberg(Matrix<T> &a){
        size_t n = a.size();
        
        Matrix<T> t = a;

        // если не ошибся (но скорее всего я ошибся), экономится n^4-3n^3+3n^2 операций умножения
        // от n^4 при стандартном алгоритме умножения матриц. Т.о. 
        // преобразование к верхней Хессенберговой форме происходит за
        // 3(n^3-n^2) = O(n^3) операций умножения

        for(size_t i = 0; i < n-2; ++i){
            std::vector<T> v(n);
            for(size_t j = 0; j < n; ++j)
                v[j] = a.data[j][i];

            Matrix<T> H = householder(v, i+1);
            // a = H * a * H;

            a = householder_mult_left(H, a, i+1);
            a = householder_mult_right(a, H, i+1);
        }
    }

    template<typename T>
    static std::complex<T> complex_eigen_value(const Matrix<T> &m, size_t i){
        T ii = m.data[i][i], jj = m.data[i+1][i+1], ij = m.data[i][i+1], ji = m.data[i+1][i];

        T b = -(ii + jj);
        T c = ii*jj - ij*ji;

        return solve_qe(T(1.0), b, c);
    }

    template<typename T>
    static bool QR_eigenV_criteria(const Matrix<T> &m, T prec, std::vector<std::complex<T>> &ev, std::vector<bool> &c, std::vector<bool> &b){
        size_t n = m.data.size();

        for(size_t i = 0; i < n-1; ++i){

            // sum of elements under subdiagonal for each column is the corresponding entry of subdiagonal since the matrix is in upper Hessenberg form
            // T val = m.data[i+1][i]; 
            // T sum = val * val;

            // sum = std::sqrt(sum);

            T sum = T(0.0);
            for(size_t j = i+1; j < n; ++j){
                T val = m.data[j][i]; 
                sum += val * val;
            }

            if(approx_equal(ev[i].imag(), T(0.0), prec) && approx_equal(sum, T(0.0), prec)){
                b[i] = true;
            }

            std::complex<T> v = complex_eigen_value(m, i);
            std::complex<T> temp = ev[i];
            ev[i] = v;

            T r1 = std::fabs(v);
            T r2 = std::fabs(temp);

            if(std::fabs(v - temp) < (r1 > r2 ? r1 : r2) * prec){
                c[i] = true;
                c[i+1] = true;
            }
        }

        for(size_t i = 0; i < n-1; ++i)
            if(approx_equal(ev[i].imag(), T(0.0), prec)){
                if(!b[i])
                    return true;
            }
            else{
                if(!c[i])
                    return true;
            }

        return false;
    }

    static long iterations_required;

public:
    template<typename T>
    static std::pair<Matrix<T>, Matrix<T>> QR_decompose(const Matrix<T> &m){
        size_t n = m.data.size();
        Matrix<T> R = m;
        Matrix<T> Q(0);

        for(size_t i = 0; i < n-1; ++i){
            std::vector<T> b(n);
            for(size_t j = 0; j < n; ++j)
                b[j] = R.data[j][i];

            Matrix<T> h = householder(b, i);
            
            // Q = Q.data.size() > 0 ? h * Q : h;
            // R = h * R;
            if(i == 0){
                Q = h;
            }
            else{
                Q = householder_mult_left(h, Q, i);
            }
            R = householder_mult_left(h, R, i);
        }

        // transposing
        for(size_t i = 1; i < n; ++i)
            for(size_t j = 0; j < i; ++j)
                std::swap(Q.data[i][j], Q.data[j][i]);

        return std::make_pair(Q, R);
    }

    template<typename T>
    static T wilkinson_shift(T a, T b, T c){
        T t = (a-c) * T(0.5);
        T bb = b*b;
        return c - sgn(t)*bb / (std::fabs(t) + std::sqrt(t*t + bb));
    }

    // Неиспользованные функции и все что закомментировано ниже - это обыкновенное "хотелось как лучше, а получилось как всегда".
    // Хессенберг, которого почти не смог применить, и сдвиги диагональных элементов должны были ускорять алгоритм (и на некоторых
    // матрицах сильно ускоряли!), как у всех нормальных людей, но что-то пошло не так, и, в общем случае, алгоритм чудовищно
    // замедлялся или вообще переставал сходиться. Но раз уж написал, то пусть валяется.
    // Хессенберга использовал очень слабо, люди пишут что можно тело основного цикла делать за куб, но ваще хз как .
    
    template<typename T>
    static std::vector<std::complex<T>> QR_eigenV(const Matrix<T> &m, T precision){
        size_t n = m.data.size();
        Matrix<T> t = m;

        Matrix<T> q(0);
        // hessenberg(t);

        std::vector<bool> c(n, false);
        std::vector<bool> b(n, false);
        std::vector<std::complex<T>> ev(n);
        iterations_required = 0;

        while(QR_eigenV_criteria(t, precision, ev, c, b)){
            // в этом блоке закомментированное относится к сдвигу

            // T sigma = t.data[n-1][n-1];
            // T sigma = wilkinson_shift(t.data[n-2][n-2], t.data[n-2][n-1], t.data[n-1][n-1]);

            // for(size_t i = 0; i < n; ++i)
            //     t.data[i][i] -= sigma;

            Matrix<T> Q(0), R(0);
            std::tie(Q, R) = EigenV::QR_decompose(t);

            t = R * Q;
            // q = q.data.size() > 0 ? q * Q : Q;

            // for(size_t i = 0; i < n; ++i)
            //     t.data[i][i] += sigma;

            ++iterations_required;
        }

        std::vector<std::complex<T>> res(n);
        for(size_t i = 0; i < n; ++i){
            if(approx_equal(ev[i].imag(), T(0.0), precision))
                res[i] = std::complex<T>(t[i][i], T(0.0));
            else{
                res[i] = ev[i];
                res[i+1] = std::complex<T>(res[i].real(), -res[i].imag());
                ++i;
            }
        }

        return res;
    }

    static long iters_required(){
        return iterations_required;
    }

    template<typename T>
    static void jacobi(Matrix<T> &mat, std::vector<T> &eigen_val, std::vector<std::vector<T>> &eigen_vec,
        T precision, bool transform_inplace=false)
    {
        if(transform_inplace){
            jacobi_helper(mat, eigen_val, eigen_vec, precision);
        }
        else{
            Matrix<T> mc = mat;
            jacobi_helper(mc, eigen_val, eigen_vec, precision);
        }
    }
};

long EigenV::iterations_required = 0;

#endif