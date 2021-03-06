CC=nvcc
CFLAGS=-Werror cross-execution-space-call -lm
DEBUGFLAGS=-g
BIN=lab1

all: main.cu error.h
	nvcc $(CFLAGS) -o $(BIN) main.cu
debug: main.cu error.h
	nvcc $(CFLAGS) $(DEBUGFLAGS) -o $(BIN) main.cu
