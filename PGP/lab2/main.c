#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include <signal.h>
#include <time.h>
#include <unistd.h>

void _sigsegv_handler(int signo) {
    printf("ERROR: segmentation fault\n");
    exit(0);
}

void _sigabrt_handler(int signo) {
    printf("ERROR: aborted\n");
    exit(0);
}

void init_error_handling() {
    signal(SIGSEGV, _sigsegv_handler);
    signal(SIGABRT, _sigabrt_handler);
}

typedef struct {
    unsigned char x, y, z, w;
} uchar4;

uchar4 avg_kernel(uchar4 *ref, int i, int j, int w, int h, int kernel_w, int kernel_h) {
    int sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
    for (int y = i; y < i + kernel_h; ++y) {
        for (int x = j; x < j + kernel_w; ++x) {
            uchar4 pixel = ref[y * w + x];
            sum_r += pixel.x;
            sum_g += pixel.y;
            sum_b += pixel.z;
            sum_a += pixel.w;
        }
    }

    int n_pixels = kernel_w * kernel_h;
    uchar4 result;
    result.x = (unsigned char)(sum_r / n_pixels);
    result.y = (unsigned char)(sum_g / n_pixels);
    result.z = (unsigned char)(sum_b / n_pixels);
    result.w = (unsigned char)(sum_a / n_pixels);

    return result;
}

void SSAA(uchar4 *ref, uchar4 *res, int w, int h, int new_w, int new_h) {
    int kernel_w = w / new_w;
    int kernel_h = h / new_h;

    for (int i = 0; i < new_h; ++i) {
        for (int j = 0; j < new_w; ++j) {
            res[i * new_w + j] = avg_kernel(ref, i * kernel_h, j * kernel_w, w, h, kernel_w, kernel_h);
        }
    }
}

int main() {
    init_error_handling();

    char *input_file_path = (char *)malloc(PATH_MAX * sizeof(char));
    char *output_file_path = (char *)malloc(PATH_MAX * sizeof(char));

    scanf("%s", input_file_path);
    scanf("%s", output_file_path);

    int new_width = 0, new_height = 0;
    scanf("%d", &new_width);
    scanf("%d", &new_height);

    FILE *input_file;
    if ((input_file = fopen(input_file_path, "rb")) == NULL) {
        printf("ERROR: can't open input file\n");
        exit(0);
    }

    free(input_file_path);

    int width = 0, height = 0;
    fread(&width, sizeof(int), 1, input_file);
    fread(&height, sizeof(int), 1, input_file);

    if (width == 0 || height == 0)
        return 0;

    int ref_n_pixels = width * height;
    int res_n_pixels = new_height * new_width;

    int ref_size = sizeof(uchar4) * ref_n_pixels;
    int res_size = sizeof(uchar4) * res_n_pixels;

    uchar4 *ref = (uchar4*)malloc(ref_size);
    uchar4 *res = (uchar4*)malloc(res_size);

    fread(ref, sizeof(uchar4), ref_n_pixels, input_file);

    fclose(input_file);

    struct timespec mt1, mt2;
    long int tt;
    clock_gettime(CLOCK_REALTIME, &mt1);

    SSAA(ref, res, width, height, new_width, new_height);

    clock_gettime(CLOCK_REALTIME, &mt2);

    tt = 1000000000 * (mt2.tv_sec - mt1.tv_sec) + (mt2.tv_nsec - mt1.tv_nsec); // ns

    fprintf(stderr, "%f\n", (double)tt / 1e6);

    free(ref);

    FILE *output_file;
    if ((output_file = fopen(output_file_path, "wb")) == NULL) {
        printf("ERROR: can't open output file\n");
        exit(0);
    }

    free(output_file_path);

    fwrite(&new_width, sizeof(int), 1, output_file);
    fwrite(&new_height, sizeof(int), 1, output_file);

    fwrite(res, sizeof(uchar4), res_n_pixels, output_file);

    fclose(output_file);

    free(res);
}