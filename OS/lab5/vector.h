#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <stdlib.h>

const int coef = 2;

typedef struct {
    int *data;
    size_t capacity;
    size_t size;
} vector;

int vector_init(vector *vec, size_t cap);
int vector_resize(vector *vec, size_t new_cap);
int vector_push_back(vector *vec, int val);
int vector_pop_back(vector *vec);
size_t vector_size(const vector *vec);
size_t vector_capacity(const vector *vec);
int vector_insert(vector *vec, size_t ind, int val);
int vector_erase(vector *vec, size_t ind);
int vector_is_empty(const vector *vec);
int vector_put(vector *vec, size_t ind, int val);
int vector_fetch(const vector *vec, size_t ind);
void vector_destroy(vector *vec);

#endif