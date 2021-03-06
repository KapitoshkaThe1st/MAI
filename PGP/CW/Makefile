CC=nvcc
NVCCFLAGS=-D_FORCE_INLINES -Werror cross-execution-space-call -O3 -lm
BIN=cw
CFLAGS=-Werror -O3 -lm

all: vec.h utils.h ssaa.h scene.h render.h object.h mat.h image.h error.h constants.h camera.h main.cu
	nvcc $(NVCCFLAGS) -o $(BIN) main.cu
