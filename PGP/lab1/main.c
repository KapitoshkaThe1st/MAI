#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <signal.h>
#include <time.h>
#include <unistd.h>

void elementwise_abs(float *vec, int n){
    for(int i = 0; i < n; ++i)
        vec[i] = fabsf(vec[i]);
}

void sigsegv_handler(int signo) {
    printf("ERROR: segmentation fault\n");
    exit(0);
}

void sigabrt_handler(int signo) {
    printf("ERROR: aborted\n");
    exit(0);
}

void init_error_handling() {
    signal(SIGSEGV, sigsegv_handler);
    signal(SIGABRT, sigabrt_handler);
}

int main() {
    init_error_handling();

    int n = 0;
    scanf("%d", &n);

    if (n == 0)
        return 0;

    int size = sizeof(float) * n;
    float *vec = (float *)malloc(size);

    for (int i = 0; i < n; ++i)
        scanf("%f", &vec[i]);

    struct timespec mt1, mt2;
    long int tt;

    clock_gettime(CLOCK_REALTIME, &mt1);

    elementwise_abs(vec, n);

    clock_gettime(CLOCK_REALTIME, &mt2);

    tt = 1000000000 * (mt2.tv_sec - mt1.tv_sec) + (mt2.tv_nsec - mt1.tv_nsec); // ns

    fprintf(stderr, "%f\n", (double)tt / 1e6);

    for (int i = 0; i < n; ++i)
        printf("%10.10e ", vec[i]);

    free(vec);
}