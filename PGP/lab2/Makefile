CC=nvcc
CFLAGS=-Werror cross-execution-space-call -lm
DEBUGFLAGS=-g
BIN=lab2

# BENCHMARK_DEFINES=--define-macro BENCHMARK --define-macro BLOCK

all: main.cu error.h benchmark.h
	nvcc $(CFLAGS) -o $(BIN) main.cu
debug: main.cu error.h benchmark.h
	nvcc $(CFLAGS) $(DEBUGFLAGS) -o $(BIN) main.cu
benchmark: main.cu error.h benchmark.h
	nvcc $(CFLAGS) --define-macro BENCHMARK --define-macro BENCHMARK -o $(BIN) main.cu
