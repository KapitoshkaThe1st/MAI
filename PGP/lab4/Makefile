CC=nvcc
CFLAGS=-Werror cross-execution-space-call -O3 -lm
DEBUGFLAGS=-g --compiler-options -g,-Wall,-Wextra,-fsanitize=address,-fsanitize=undefined,-fsanitize=leak,-fsanitize=pointer-subtract,-fsanitize=pointer-compare,-fno-omit-frame-pointer,-fstack-protector-all,-fstack-clash-protection,-fcf-protection
BIN=lab4

all: main.cu error.h
	nvcc $(CFLAGS) -o $(BIN) main.cu

debug: main.cu error.h benchmark.h
	nvcc $(CFLAGS) $(DEBUGFLAGS) -o $(BIN) main.cu
benchmark: main.cu error.h benchmark.h
	nvcc $(CFLAGS) --define-macro BENCHMARK --define-macro GRID_DIM=1024 --define-macro BLOCK_DIM=256 -o $(BIN) main.cu
