HEADERS	= ./headers
KERNELS = ./kernels
VPATH = $(HEADERS):$(KERNELS)
OBJS = main.o cuda_wrapper.o update.o calc_acc.o

NVCC = nvcc
CXX = g++
MPICC = mpicc

CUDA_PATH ?= "/usr/local/cuda-7.5/lib64"
NVCCFLAGS = -ccbin=$(CXX) -Xcompiler
CXXFLAGS = -O3 -I. -I$(HEADERS) -D_FORCE_INLINES
CFLAGS = -std=c99
LDFLAGS = -lcudart -L/opt/apps/cuda/6.5/lib64/ -L$(CUDA_PATH) -lm

EXEC = nbody

all: $(EXEC)

update.o: update.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $<

calc_acc.o: calc_acc.cu
		$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $<

cuda_wrapper.o: cuda_wrapper.cu kernels.cuh
		$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<

main.o :main.c
	$(MPICC) $(CXXFLAGS) $(CFLAGS) -c $<

$(EXEC): $(OBJS)
	$(MPICC) $(CXXFLAGS)  $^  -o $@ $(LDFLAGS)

clean:
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
