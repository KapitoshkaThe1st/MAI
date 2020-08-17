#ifndef TDMA_H
#define TDMA_H

#include <vector>
#include <cmath>
#include <limits>

#include "eps.h"
#include "math_utils.h"

std::vector<double> tdma(const std::vector<double> &a, const std::vector<double> &b,
     const std::vector<double> &c, const std::vector<double> &d, size_t n){
         
    std::vector<double> p(n - 1);
    std::vector<double> q(n);

    p[0] = -c[0] / b[0];
    q[0] = d[0] / b[0];

    for(size_t i = 1; i < n; ++i){
        double denum = a[i-1] * p[i-1] + b[i];

        if(i < n - 1)
            p[i] = -c[i] / denum; 
        q[i] = (d[i] - a[i-1] * q[i-1]) / denum;
    }

    std::vector<double> res(n);

    res[n-1] = q[n-1];

    for (size_t i = n - 2; i != std::numeric_limits<size_t>::max(); --i) {
        double x = p[i] * res[i+1] + q[i];
        res[i] = x;
    }

    return res;
}

#endif
