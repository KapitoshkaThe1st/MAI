#include <stdio.h>

#include "vector.h"

int main(){
    printf("init time linked dynamic library interaction\n");

    vector v;
    printf("vector initialization\n");
    vector_init(&v, 4);

    printf("vector cap: %ld\n", vector_capacity(&v));
    printf("vector size: %ld\n", vector_size(&v));

    printf("push 3 elements\n");
    vector_push_back(&v, 1);
    vector_push_back(&v, 2);
    vector_push_back(&v, 3);

    printf("vector cap: %ld\n", vector_capacity(&v));
    printf("vector size: %ld\n", vector_size(&v));

    printf("push extra 2 elements\n");
    vector_push_back(&v, 2);
    vector_push_back(&v, 3);

    printf("vector cap: %ld\n", vector_capacity(&v));
    printf("vector size: %ld\n", vector_size(&v));

    printf("pop 3 elements\n");
    vector_pop_back(&v);
    vector_pop_back(&v);
    vector_pop_back(&v);

    printf("vector cap: %ld\n", vector_capacity(&v));
    printf("vector size: %ld\n", vector_size(&v));

    printf("vector contains: ");
    size_t size = vector_size(&v);
    for(size_t i = 0; i < size; ++i)
        printf("%d ", vector_fetch(&v, i));
    printf("\n");

    printf("insert 100 on 1th position in vector\n");
    vector_insert(&v, 1, 100);

    printf("vector contains: ");
    size = vector_size(&v);
    for (size_t i = 0; i < size; ++i)
        printf("%d ", vector_fetch(&v, i));
    printf("\n");

    printf("erase 0th element in vector\n");
    vector_erase(&v, 0);

    printf("vector contains: ");
    size = vector_size(&v);
    for (size_t i = 0; i < size; ++i)
        printf("%d ", vector_fetch(&v, i));
    printf("\n");

    printf("put 111 on 0th position in vector\n");
    vector_put(&v, 0, 111);

    printf("vector contains: ");
    size = vector_size(&v);
    for (size_t i = 0; i < size; ++i)
        printf("%d ", vector_fetch(&v, i));
    printf("\n");

    printf("vector destroying\n");
    vector_destroy(&v);

    return 0;
}