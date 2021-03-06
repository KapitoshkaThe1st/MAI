CC=nvcc
CFLAGS=-Werror cross-execution-space-call -lm
# DEBUGFLAGS=-g --compiler-options -g,-Wall,-Wextra,-fsanitize=address,-fsanitize=undefined
DEBUGFLAGS=-g --compiler-options -g,-Wall,-Wextra,-fsanitize=address,-fsanitize=undefined,-fsanitize=leak,-fsanitize=pointer-subtract,-fsanitize=pointer-compare,-fno-omit-frame-pointer,-fstack-protector-all,-fstack-clash-protection,-fcf-protection
BIN=l3

all: main.cu error.h benchmark.h
	nvcc $(CFLAGS) -o $(BIN) main.cu
debug: main.cu error.h benchmark.h
	nvcc $(CFLAGS) $(DEBUGFLAGS) -o $(BIN) main.cu
benchmark: main.cu error.h benchmark.h
	nvcc $(CFLAGS) --define-macro BENCHMARK --define-macro BENCHMARK -o $(BIN) main.cu
