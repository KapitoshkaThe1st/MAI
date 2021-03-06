#ifndef MAT_H
#define MAT_H

#include "vec.h"
#include <iostream>

struct mat3{
    float m[3][3];
    __host__ __device__ mat3() {}
    __host__ __device__ mat3(float m11, float m12, float m13,
            float m21, float m22, float m23,
            float m31, float m32, float m33){
        m[0][0] = m11; m[0][1] = m12; m[0][2] = m13;
        m[1][0] = m21; m[1][1] = m22; m[1][2] = m23;
        m[2][0] = m31; m[2][1] = m32; m[2][2] = m33;
    }

    __host__ __device__ static mat3 identity(){
        mat3 res;
        for(int i = 0; i < 3; ++i){
            for(int j = 0; j < 3; ++j){
                res.m[i][j] = 0.0f;
            }
        }
        for(int i = 0; i < 3; ++i)
            res.m[i][i] = 1.0f;

        return res;
    } 

};

std::ostream& operator<<(std::ostream &os, const mat3 &m){
    for(int i = 0 ; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            os << m.m[i][j] << ' ';
        }
        os << std::endl;
    }
    return os;
}

__host__ __device__ float det(const mat3 &m){
    return m.m[0][0] * m.m[1][1] * m.m[2][2] + m.m[1][0] * m.m[0][2] * m.m[2][1] + m.m[2][0] * m.m[0][1] * m.m[1][2]
        - m.m[0][2] * m.m[1][1] * m.m[2][0] - m.m[0][0] * m.m[1][2] * m.m[2][1] - m.m[0][1] * m.m[1][0] * m.m[2][2]; 
}

__host__ __device__ mat3 inv(const mat3 &m){
        float d = det(m);

        float m11 = (m.m[1][1] * m.m[2][2] - m.m[2][1] * m.m[1][2]) / d;
        float m12 = (m.m[2][1] * m.m[0][2] - m.m[0][1] * m.m[2][2]) / d;
        float m13 = (m.m[0][1] * m.m[1][2] - m.m[1][1] * m.m[0][2]) / d;

        float m21 = (m.m[2][0] * m.m[1][2] - m.m[1][0] * m.m[2][2]) / d;
        float m22 = (m.m[0][0] * m.m[2][2] - m.m[2][0] * m.m[0][2]) / d;
        float m23 = (m.m[1][0] * m.m[0][2] - m.m[0][0] * m.m[1][2]) / d;

        float m31 = (m.m[1][0] * m.m[2][1] - m.m[2][0] * m.m[1][1]) / d;
        float m32 = (m.m[2][0] * m.m[0][1] - m.m[0][0] * m.m[2][1]) / d;
        float m33 = (m.m[0][0] * m.m[1][1] - m.m[1][0] * m.m[0][1]) / d;

        return mat3(m11, m12, m13,
                    m21, m22, m23,
                    m31, m32, m33);
    }

struct mat4{
    float m[4][4];
    __host__ __device__ mat4() {};
    __host__ __device__ mat4(float m11, float m12, float m13, float m14,
            float m21, float m22, float m23, float m24,
            float m31, float m32, float m33, float m34,
            float m41, float m42, float m43, float m44)
    {
        m[0][0] = m11; m[0][1] = m12; m[0][2] = m13; m[0][3] = m14;
        m[1][0] = m21; m[1][1] = m22; m[1][2] = m23; m[1][3] = m24;
        m[2][0] = m31; m[2][1] = m32; m[2][2] = m33; m[2][3] = m34;
        m[3][0] = m41; m[3][1] = m42; m[3][2] = m43; m[3][3] = m44;
    }
};

std::ostream& operator<<(std::ostream &os, const mat4 &m){
    for(int i = 0 ; i < 4; ++i){
        for(int j = 0; j < 4; ++j){
            os << m.m[i][j] << ' ';
        }
        os << std::endl;
    }
    return os;
}

__host__ __device__ mat3 operator*(const mat3 &a, const mat3 &b){
    mat3 res;
    for(int i = 0; i < 3; ++i){
        for(int j = 0; j < 3; ++j){
            float sum = 0.0f;
            for(int k = 0; k < 3; ++k){
                sum += a.m[i][k] * b.m[k][j];
            }
            res.m[i][j] = sum;
        }
    }
    return res;
}

__host__ __device__ mat3 operator+(const mat3 &a, const mat3 &b){
    mat3 res;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            res.m[i][j] = a.m[i][j] + b.m[i][j];
    
    return res;
}

__host__ __device__ mat3 operator*(float a, const mat3 &m){
    mat3 res;
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
            res.m[i][j] = a * m.m[i][j];
    
    return res;
}

__host__ __device__ mat3 operator*(const mat3 &m, float a){
    return a * m;
}

__host__ __device__ vec3 operator*(const mat3 &m, const vec3 &v){
    vec3 res;
    res.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z; 
    res.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z; 
    res.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z; 
    return res;
}

__host__ __device__ vec4 operator*(const mat4 &m, const vec4 &v){
    vec4 res;
    res.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w; 
    res.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w; 
    res.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w;
    res.w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w; 
    return res;
}

__host__ __device__ vec3 homogeneous_mult(const mat4 &m, const vec3 &v){
    vec4 tmp(v.x, v.y, v.z, 1.0f);
    tmp = m * tmp;
    return {tmp.x, tmp.y, tmp.z};
}

__host__ __device__ mat3 align_mat(const vec3 &a, const vec3 &b){
    vec3 v = cross_product(a, b);
    float c = dot_product(a, b);

    mat3 m(0.0f, -v.z, v.y,
           v.z, 0.0f, -v.x,
           -v.y, v.x, 0.0f);

    return mat3::identity() + m + 1.0f / (1.0f + c) * m * m;
}

#endif