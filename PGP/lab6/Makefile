CC=nvcc
CFLAGS=-Werror cross-execution-space-call -lm -lcublas -lcurand -lGL -lGLU -lglut -lGLEW

DEBUGFLAGS=-g --compiler-options -g,-Wall,-Wextra,-fsanitize=address,-fsanitize=undefined,-fsanitize=leak,-fsanitize=pointer-subtract,-fsanitize=pointer-compare,-fno-omit-frame-pointer,-fstack-protector-all,-fstack-clash-protection,-fcf-protection
BIN=lab6

all: main.cu error.h
	nvcc $(CFLAGS) -o $(BIN) main.cu
