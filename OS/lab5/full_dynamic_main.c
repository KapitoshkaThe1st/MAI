#include <stdio.h>
#include <dlfcn.h>

#include "vector.h"

int (*vec_init)(vector *vec, size_t cap);
int (*vec_resize)(vector *vec, size_t new_cap);
int (*vec_push_back)(vector *vec, int val);
int (*vec_pop_back)(vector *vec);
size_t (*vec_size)(const vector *vec);
size_t (*vec_capacity)(const vector *vec);
int (*vec_insert)(vector *vec, size_t ind, int val);
int (*vec_erase)(vector *vec, size_t ind);
int (*vec_is_empty)(const vector *vec);
int (*vec_put)(vector *vec, size_t ind, int val);
int (*vec_fetch)(const vector *vec, size_t ind);
void (*vec_destroy)(vector *vec);

void *dl_handler;

void dl_init(){
    dl_handler = dlopen("libvector.so", RTLD_LAZY);
    if(!dl_handler){
        fprintf(stderr, "ERROR: %s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    vec_init = dlsym(dl_handler, "vector_init");
    vec_resize = dlsym(dl_handler, "vector_resize");
    vec_push_back = dlsym(dl_handler, "vector_push_back");
    vec_pop_back = dlsym(dl_handler, "vector_pop_back");
    vec_size = dlsym(dl_handler, "vector_size");
    vec_capacity = dlsym(dl_handler, "vector_capacity");
    vec_insert = dlsym(dl_handler, "vector_insert");
    vec_erase = dlsym(dl_handler, "vector_erase");
    vec_is_empty = dlsym(dl_handler, "vector_is_empty");
    vec_put = dlsym(dl_handler, "vector_put");
    vec_fetch = dlsym(dl_handler, "vector_fetch");
    vec_destroy = dlsym(dl_handler, "vector_destroy");
}

void dl_fini(){
    dlclose(dl_handler);
}

int main(){
    printf("fully dynamic linked dynamic library interaction\n");
    dl_init();
    vector v;
    printf("vector initialization\n");
    (*vec_init)(&v, 4);

    printf("vector cap: %ld\n", (*vec_capacity)(&v));
    printf("vector size: %ld\n", (*vec_size)(&v));

    printf("push 3 elements\n");
    (*vec_push_back)(&v, 1);
    (*vec_push_back)(&v, 2);
    (*vec_push_back)(&v, 3);

    printf("vector cap: %ld\n", (*vec_capacity)(&v));
    printf("vector size: %ld\n", (*vec_size)(&v));

    printf("push extra 2 elements\n");
    (*vec_push_back)(&v, 2);
    (*vec_push_back)(&v, 3);

    printf("vector cap: %ld\n", (*vec_capacity)(&v));
    printf("vector size: %ld\n", (*vec_size)(&v));

    printf("pop 3 elements\n");
    (*vec_pop_back)(&v);
    (*vec_pop_back)(&v);
    (*vec_pop_back)(&v);

    printf("vector cap: %ld\n", (*vec_capacity)(&v));
    printf("vector size: %ld\n", (*vec_size)(&v));

    printf("vector contains: ");
    size_t size = (*vec_size)(&v);
    for (size_t i = 0; i < size; ++i)
        printf("%d ", (*vec_fetch)(&v, i));
    printf("\n");

    printf("insert 100 on 1th position in vector\n");
    (*vec_insert)(&v, 1, 100);

    printf("vector contains: ");
    size = (*vec_size)(&v);
    for (size_t i = 0; i < size; ++i)
        printf("%d ", (*vec_fetch)(&v, i));
    printf("\n");

    printf("erase 0th element in vector\n");
    (*vec_erase)(&v, 0);

    printf("vector contains: ");
    size = (*vec_size)(&v);
    for (size_t i = 0; i < size; ++i)
        printf("%d ", (*vec_fetch)(&v, i));
    printf("\n");

    printf("put 111 on 0th position in vector\n");
    (*vec_put)(&v, 0, 111);

    printf("vector contains: ");
    size = (*vec_size)(&v);
    for (size_t i = 0; i < size; ++i)
        printf("%d ", (*vec_fetch)(&v, i));
    printf("\n");

    printf("vector destroying\n");
    (*vec_destroy)(&v);

    dl_fini();
    return 0;
}