CC=nvcc
CFLAGS=-D_FORCE_INLINES -Werror cross-execution-space-call -O3 -lm
DEBUGFLAGS=-g --compiler-options -g,-Wall,-Wextra,-fsanitize=address,-fsanitize=undefined,-fsanitize=leak,-fsanitize=pointer-subtract,-fsanitize=pointer-compare,-fno-omit-frame-pointer,-fstack-protector-all,-fstack-clash-protection,-fcf-protection
BIN=lab5

all: main.cu error.h
	nvcc $(CFLAGS) -o $(BIN) main.cu

