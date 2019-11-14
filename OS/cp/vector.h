#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

const int coef = 2;

typedef struct {
    char *data;
    size_t capacity;
    size_t size;
    size_t type_size;
} vector;

#define vector_init(VEC, CAP, TYPE) vector_ini(VEC, CAP, sizeof(TYPE))

void _put(vector *vec, size_t ind, void *val){
    memcpy(vec->data + ind * vec->type_size, val, vec->type_size);
}

void* _get(const vector *vec, size_t ind){
    return vec->data + ind * vec->type_size;
}

int vector_ini(vector *vec, size_t cap, size_t ts){
    vec->type_size = ts;
    vec->data = (char *)malloc(vec->type_size * cap);
    if (!vec->data) 
        return 0;
    vec->capacity = cap;
    vec->size = 0;
    return 1;
}

int vector_resize(vector *vec, size_t new_cap){
    vec->capacity = new_cap;
    vec->data = (char *)realloc(vec->data, vec->capacity * vec->type_size);
    if (!vec->data)
        return 0;
    return 1;
}

int vector_push_back(vector *vec, void* val){
    if(vec->size >= vec->capacity)
        if(!vector_resize(vec, vec->capacity * coef))
            return 0;
    _put(vec, vec->size, val);
    ++vec->size;
    return 1;
}

int vector_pop_back(vector *vec){
    --vec->size;
    if(vec->size * coef * coef <= vec->capacity)
        if(!vector_resize(vec, vec->capacity / coef))
            return 0;
    return 1;
}

int vector_push_front(vector *vec, void *val){
    if(vec->size >= vec->capacity)
        if(!vector_resize(vec, vec->capacity * coef))
            return 0;
    memmove(vec->data + vec->type_size, vec->data, vec->size * vec->type_size);
    _put(vec, 0, val);
    vec->size++;
    return 1;
}

int vector_pop_front(vector *vec){
    --vec->size;
    if(vec->size * coef * coef <= vec->capacity)
        if(!vector_resize(vec, vec->capacity / coef))
            return 0;
    memmove(vec->data, vec->data + vec->type_size, vec->size * vec->type_size);
    return 1;
}

size_t vector_size(const vector *vec){
    return vec->size;
}

size_t vector_capacity(const vector *vec){
    return vec->capacity;
}

int vector_insert(vector *vec, size_t ind, void *val){
    if (ind < 0 || ind > vec->size)
        return 0;
    if(vec->size + 1 > vec->capacity)
        if(!vector_resize(vec, vec->capacity * coef))
            return 0;
    for(size_t i = vec->size; i > ind; --i)
        _put(vec, i, _get(vec, i-1));
    _put(vec, ind, val);
    ++vec->size;

    return 1;
}

int vector_erase(vector *vec, size_t ind){
    if (ind < 0 || ind > vec->size)
        return 0;
    for (size_t i = ind + 1; i < vec->size; ++i)
        _put(vec, i-1, _get(vec, i));
    --vec->size;
    if (vec->size * coef * coef <= vec->capacity)
        if(!vector_resize(vec, vec->capacity / coef))
            return 0;
    return 1;
}

int vector_is_empty(const vector *vec){
    return vec->size == 0;
}

int vector__put(vector *vec, size_t ind, void *val){
    if(ind < 0 || ind >= vec->size)
        return 0;
    _put(vec, ind, val);
    return 1;
}

int vector_fetch(const vector *vec, size_t ind, void *val){
    if (ind < 0 || ind >= vec->size)
        return 0;
    memcpy(val, _get(vec, ind), vec->type_size);
    return 1;
}

void vector_back(const vector *vec, void *val){
    vector_fetch(vec, vec->size - 1, val);
}

void vector_front(const vector *vec, void *val){
    vector_fetch(vec, 0, val);
}

void vector_destroy(vector *vec){
    free(vec->data);
    vec->capacity = 0;
    vec->size = 0;
    vec->data = NULL;
    vec->type_size = 0;
}

#endif