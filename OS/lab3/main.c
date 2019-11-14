#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>

#define THREADS_LIMIT   8      // максимальное количество потоков по умолчанию (оптимальное)
#define STR_SIZE        128    // длина строки

int string_compare_nullable(char *str1, char *str2);    // сравнение строк. Если одна из строк NULL, то она принимается большей
int next_2pow(int n);                                   // получение следующей степени двойки
void shuffle(char **arr, int l, int r);                 // перетасовка (подсобная функция для слияния Бэтчера)
void unshuffle(char **arr, int l, int r);               // обратная перетасовка (подсобная функция для слияния Бэтчера)
void comp_exch_str(char **a, char **b);                 // компаратор строк
void batcher_merge(char **arr, int l, int m, int r);    // четно-нечетное слияние Бэтчера
void* merge_sort_parallel(void *args);                  // обертка над сортировкой слиянием для выполнения в отдельном потоке
void merge_sort(char **arr, int l, int r);              // сортировка слиянием

int thread_count = 0;                   // количество работающих потоков
pthread_mutex_t thread_count_locker;    // для корректного взаимодействия со счетчиком потоков
int threads_limit = THREADS_LIMIT;      // максимальное количество потоков

int main(int argc, char const *argv[])
{
    /* Получение ограничения по потокам */
    if(argc == 3){
        if(strcmp(argv[1], "-t") == 0){
            threads_limit = atoi(argv[2]);
        }
    }
    /* Инициализация */
    int status = 0;
    status = pthread_mutex_init(&thread_count_locker, NULL);
    if(status){
        fprintf(stderr, "ERROR: mutex initialization\n");
        return EXIT_FAILURE;
    }

    /* Считываем строки в импровизированный вектор */
    int capacity = 1;
    int size = 0;
    char **arr = (char **)malloc(sizeof(char *) * capacity);
    if(!arr){
        fprintf(stderr, "ERROR: bad allocation\n");
        return EXIT_FAILURE;
    }
    while (1)
    {
        arr[size] = (char *)malloc(sizeof(char) * (STR_SIZE + 1));
        if(!arr[size]){
            fprintf(stderr, "ERROR: bad allocation\n");
            return EXIT_FAILURE;
        }
        if (scanf("%s", arr[size]) != 1){
            free(arr[size]);
            break;
        }
        ++size;
        if (size == capacity){
            capacity *= 2;
            arr = (char **)realloc(arr, sizeof(char *) * capacity);
            if (!arr)
            {
                fprintf(stderr, "ERROR: bad allocation\n");
                return EXIT_FAILURE;
            }
        }
    }

    /* Заполняем пустые места NULL'ами. При сравнении со строкой он всегда будет больше */
    for (int i = size; i < capacity; ++i)
        arr[i] = NULL;

    merge_sort(arr, 0, (int)pow(2, next_2pow(size)) - 1);

    /* Освобождаем ресурсы */
    for (int i = 0; i < size; ++i)
        printf("%s ", arr[i]);
    printf("\n");
    for(int i = 0; i < size; ++i){
        free(arr[i]);
    }
    free(arr);
    pthread_mutex_destroy(&thread_count_locker);

    return 0;
}

int string_compare_nullable(char *str1, char *str2){
    if(str1 == NULL)
        return 1;
    else if(str2 == NULL)
        return -1;
    else
        return strcmp(str1, str2);    
}

int next_2pow(int n)
{
    double l = log2(n);
    if (l - (double)(int)(l) == 0.0)
        return (int)l;
    else
        return (int)l + 1;
}

void shuffle(char **arr, int l, int r)
{
    int count = r - l + 1;
    int m = (r + l) / 2;
    char **temp = (char **)malloc(count * sizeof(char*));
    for (int i = 0, k = 0; i + l <= m; ++i, k += 2)
    {
        temp[k] = arr[l + i];
        temp[k + 1] = arr[m + i + 1];
    }
    for (int i = 0; i < count; ++i)
        arr[l + i] = temp[i];
    free(temp);
}

void unshuffle(char **arr, int l, int r)
{
    int count = r - l + 1;
    int m = count / 2;
    char **temp = (char **)malloc(count * sizeof(char*));
    for (int i = 0, k = l; k < r; ++i, k += 2)
    {
        temp[i] = arr[k];
        temp[i + m] = arr[k + 1];
    }
    for (int i = 0; i < count; ++i)
        arr[l + i] = temp[i];
    free(temp);
}

void comp_exch_str(char **a, char **b)
{
    if (string_compare_nullable(*a, *b) > 0)
    {
        char* temp = *a;
        *a = *b;
        *b = temp;
    }
}

void batcher_merge(char **arr, int l, int m, int r)
{
    if (l + 1 == r)
    {
        comp_exch_str(&arr[l], &arr[r]);
    }
    if (l + 2 > r)
        return;
    unshuffle(arr, l, r);
    batcher_merge(arr, l, (l + m) / 2, m);
    batcher_merge(arr, m + 1, (m + r + 1) / 2, r);
    shuffle(arr, l, r);
    for (int i = l + 1; i < r; i += 2)
    {
        comp_exch_str(&arr[i], &arr[i + 1]);
    }
}

typedef struct
{
    char **arr;
    int l, r;
} sort_data;

void *merge_sort_parallel(void *args)
{
    sort_data *data = (sort_data *)args;
    merge_sort(data->arr, data->l, data->r);
    return NULL;
}

void merge_sort(char **arr, int l, int r){
    int m = (r + l) / 2;
    if (l == r)
        return;

    // fprintf(stderr, "thread count: %d\n", thread_count);

    int is_threaded = 0;
    pthread_t thread_left;
    pthread_t thread_right;
    int thread_available = 0;
    int status = 0;
    /* Для корректоного получения/изменения значения переменной из разных потоков */
    pthread_mutex_lock(&thread_count_locker);
    if(thread_count < threads_limit){
        thread_available = threads_limit - thread_count;
        if(thread_available >= 2)
            thread_count += 2;
        else if(thread_available == 1)
            ++thread_count;
    }
    pthread_mutex_unlock(&thread_count_locker);

    if (thread_count > threads_limit)
    {
        fprintf(stderr, "ERROR: thread limit exceeded: %d\n", thread_count);
        return;
    }
    /* Если доступно 2 и больше потоков */
    if (thread_available >= 2)
    {
        // fprintf(stderr, "thread count: %d\n", thread_count);
        is_threaded = 1;
        sort_data data_left = {arr, l, m};
        sort_data data_right = {arr, m + 1, r};

        /* вызываем параллельно если потоков не больше, чем огнаничение */
        status = pthread_create(&thread_left, NULL, merge_sort_parallel, &data_left);
        if(status){
            fprintf(stderr, "ERROR: thread creation failed\n");
            return;
        }
        status = pthread_create(&thread_right, NULL, merge_sort_parallel, &data_right);
        if (status)
        {
            fprintf(stderr, "ERROR: thread creation failed\n");
            return;
        }
    }
    /* Если доступен 1 поток */
    else if(thread_available == 1){
        is_threaded = 1;
        sort_data data_left = {arr, l, m};
        status = pthread_create(&thread_left, NULL, merge_sort_parallel, &data_left);
        if (status)
        {
            fprintf(stderr, "ERROR: thread creation failed\n");
            return;
        }
        merge_sort(arr, m + 1, r);
    }
    /* Если нет доступных потоков, то мучаемся сами */
    else{
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);
    }

    /* синхронизируем */
    if (is_threaded){
        /* если было создано 2 потока */
        if(thread_available >= 2){
            status = pthread_join(thread_left, NULL);
            if (status)
            {
                fprintf(stderr, "ERROR: thread joining failed\n");
                return;
            }
            status = pthread_join(thread_right, NULL);
            if (status)
            {
                fprintf(stderr, "ERROR: thread joining failed\n");
                return;
            }
            /* если был создан 1*/
        }else if(thread_available == 1){
            status = pthread_join(thread_left, NULL);
            if (status)
            {
                fprintf(stderr, "ERROR: thread joining failed\n");
                return;
            }
        }
        /* Для корректоного получения/изменения значения переменной из разных потоков */
        /* "возвращаем" занимаемые потоками места */
        pthread_mutex_lock(&thread_count_locker);
        if(thread_available >= 2)
            thread_count -= 2;
        else if(thread_available == 1)
            thread_count--;
        pthread_mutex_unlock(&thread_count_locker);
    }

    batcher_merge(arr, l, m, r);
}
