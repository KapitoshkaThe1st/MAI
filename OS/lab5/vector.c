#include "vector.h"

int vector_init(vector *vec, size_t cap){
    vec->data = (int *)malloc(sizeof(int) * cap);
    if (!vec->data) 
        return 0;
    vec->capacity = cap;
    vec->size = 0;
    return 1;
}

int vector_resize(vector *vec, size_t new_cap){
    vec->capacity = new_cap;
    vec->data = (int *)realloc(vec->data, vec->capacity * sizeof(int));
    if (!vec->data)
        return 0;
    return 1;
}

int vector_push_back(vector *vec, int val){
    if(vec->size >= vec->capacity)
        if(!vector_resize(vec, vec->capacity * coef))
            return 0;
    vec->data[vec->size] = val;
    ++vec->size;
    return 1;
}

int vector_pop_back(vector *vec){
    --vec->size;
    if (vec->size * coef * coef <= vec->capacity)
        if(!vector_resize(vec, vec->capacity / coef))
            return 0;
}

size_t vector_size(const vector *vec){
    return vec->size;
}

size_t vector_capacity(const vector *vec){
    return vec->capacity;
}

int vector_insert(vector *vec, size_t ind, int val){
    if (ind < 0 || ind > vec->size)
        return 0;
    if(vec->size + 1 > vec->capacity)
        if(!vector_resize(vec, vec->capacity * coef))
            return 0;
    for(size_t i = vec->size; i > ind; --i)
        vec->data[i] = vec->data[i-1];
    vec->data[ind] = val;
    ++vec->size;

    return 1;
}

int vector_erase(vector *vec, size_t ind){
    if (ind < 0 || ind > vec->size)
        return 0;
    for (size_t i = ind + 1; i < vec->size; ++i)
        vec->data[i-1] = vec->data[i];
    --vec->size;
    if (vec->size * coef * coef <= vec->capacity)
        if(!vector_resize(vec, vec->capacity / coef))
            return 0;
    return 1;
}

int vector_is_empty(const vector *vec){
    return vec->size == 0;
}

int vector_put(vector *vec, size_t ind, int val){
    if(ind < 0 || ind >= vec->size)
        return 0;
    vec->data[ind] = val;
    return 1;
}

int vector_fetch(const vector *vec, size_t ind){
    if (ind < 0 || ind >= vec->size)
        return -1;
    return vec->data[ind];
}

void vector_destroy(vector *vec){
    free(vec->data);
    vec->capacity = 0;
    vec->size = 0;
    vec->data = NULL;
}