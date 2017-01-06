HEADERS	= ./headers
KERNELS = ./kernels
VPATH = $(HEADERS):$(KERNELS)
OBJS = main.o cuda_wrapper.o update.o calc_acc.o

NVCC = nvcc
CXX = g++
MPICXX = mpicc

CUDA_PATH ?= "/usr/local/cuda-7.5/lib64"
NVCCFLAGS = -ccbin=$(CXX) -Xcompiler -arch=compute_35 -code=sm_35
CXXFLAGS = -O3 -I. -I$(HEADERS) -D_FORCE_INLINES
LDFLAGS = -lcudart -L/opt/apps/cuda/6.5/lib64/lcudart -L$(CUDA_PATH) -lm

EXEC = nbody

all: $(EXEC)

update.o: update.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
calc_acc.o: calc_acc.cu
		$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<

cuda_wrapper.o: cuda_wrapper.cu kernels.cuh
		$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<

main.o :main.c
	$(MPICXX) $(CXXFLAGS) $(NVFLAGS) -c $<

$(EXEC): $(OBJS)
	$(MPICXX) $(CXXFLAGS)  $^  -o $@ $(LDFLAGS)

clean:
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
