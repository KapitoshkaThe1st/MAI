#ifndef UTILS_H
#define UTILS_H

#include <chrono>

#define RUNTIME_ERROR(X) std::runtime_error(std::string(X) + __FILE__ ":" + std::to_string(__LINE__));

#define BENCHMARK(Y, X)                                                                                              \
    {                                                                                                                \
        auto start = std::chrono::steady_clock::now();                                                               \
        X                                                                                                            \
        auto end = std::chrono::steady_clock::now();                                                                 \
        auto diff = end - start;                                                                                     \
        std::cout << Y << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " ms" << std::endl; \
    }

#endif