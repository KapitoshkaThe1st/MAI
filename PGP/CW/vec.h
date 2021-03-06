#ifndef VEC_H
#define VEC_H

#include <iostream>
#include <cmath>

struct vec3{
    float x, y, z;
    __host__ __device__ vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ vec3(float val) : x(val), y(val), z(val) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
} __attribute__ ((aligned(16)));

__host__ __device__ float len(vec3 v){
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ vec3 normalize(vec3 v){
    float l = len(v);
    return {v.x / l, v.y / l, v.z / l};
}

__host__ __device__ vec3 operator+(const vec3 &a, const vec3 &b){
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ void operator+=(vec3 &a, const vec3 &b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__host__ __device__ vec3 operator-(const vec3 &a){
    return {-a.x, -a.y, -a.z};
}

__host__ __device__ vec3 operator-(const vec3 &a, const vec3 &b){
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ void operator-=(vec3 &a, const vec3 &b){
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__host__ __device__ vec3 operator*(float c, const vec3 &v){
    return {c * v.x, c * v.y, c * v.z};
}

__host__ __device__ vec3 operator*(const vec3 &v, float c){
    return c * v;
}

__host__ __device__ vec3 operator*(const vec3 &v, const vec3 &w){
    return {v.x * w.x, v.y * w.y, v.z * w.z};
}

std::ostream& operator<<(std::ostream &os, const vec3 &f){
    os << f.x << ' ' << f.y << ' ' << f.z;
    return os;
}

std::istream& operator>>(std::istream &is, vec3 &f){
    is >> f.x >> f.y >> f.z;
    return is;
}

struct vec4 {
    float x, y, z, w;
    __host__ __device__ vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    __host__ __device__ vec4(float val) : x(val), y(val), z(val), w(val) {}
    __host__ __device__ vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
} __attribute__ ((aligned(16)));

__host__ __device__ float len(vec4 v){
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
}

__host__ __device__ vec4 normalize(vec4 v){
    float l = len(v);
    return {v.x / l, v.y / l, v.z / l, v.w / l};
}

__host__ __device__ vec4 operator+(const vec4 &a, const vec4 &b){
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__host__ __device__ void operator+=(vec4 &a, const vec4 &b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__host__ __device__ vec4 operator-(const vec4 &a){
    return {-a.x, -a.y, -a.z, -a.w};
}

__host__ __device__ vec4 operator-(const vec4 &a, const vec4 &b){
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

__host__ __device__ void operator-=(vec4 &a, const vec4 &b){
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

__host__ __device__ vec4 operator*(float c, const vec4 &v){
    return {c * v.x, c * v.y, c * v.z, c * v.w};
}

__host__ __device__ vec4 operator*(const vec4 &v, const vec4 &w){
    return {v.x * w.x, v.y * w.y, v.z * w.z, v.w * w.w};
}

__host__ __device__ vec4 operator*(const vec4 &v, float c){
    return c * v;
}

std::ostream& operator<<(std::ostream &os, const vec4 &f){
    os << f.x << ' ' << f.y << ' ' << f.z << ' ' << f.w;
    return os;
}

std::istream& operator>>(std::istream &is, vec4 &f){
    is >> f.x >> f.y >> f.z >> f.w;
    return is;
}

__host__ __device__ vec3 cross_product(const vec3 &a, const vec3 &b){
    vec3 res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;

    return res;
}

__host__ __device__ float dot_product(const vec3 &a, const vec3 &b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ vec3 reflect(const vec3 vec, const vec3 normal){
    return vec - 2.0f * dot_product(vec, normal) * normal;
}

__host__ __device__ vec3 refract(const vec3 &vec, const vec3 &normal, float n1, float n2){
    float r = n1 / n2;
    float c = -dot_product(normal, vec);

    return r * vec + (r * c - sqrt(1.0f - r*r * (1.0f - c*c))) * normal;
}

#endif