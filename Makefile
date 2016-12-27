HEADERS	= ./headers
KERNELS = ./kernels
NVCC = nvcc
MPICXX = mpicxx

CUDA_PATH ?= "/usr/local/cuda-8.0"

NVCCFLAGS = -ccbin=$(CXX) -Xcompiler
CXXFLAGS = -O3 -I. -I$(HEADERS) -D_FORCE_INLINES
LIBS = -lcudart -L/opt/apps/cuda/6.5/lib64/lcudart -L/usr/local/cuda-8.0/lib64/

EXEC = nbody-cuda

all: $(EXEC)



# Load common make options
LDFLAGS	= -lcudart -L/opt/apps/cuda/6.5/lib64/lcudart -L/usr/local/cuda-8.0/lib64/

update.o: $(KERNELS)/update.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
calc_acc.o: $(KERNELS)/calc_acc.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
wrapper.o: wrapper.cu
		$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<

nbody-cuda.o :nbody-cuda.cpp
	$(MPICXX) $(CXXFLAGS) $(NVFLAGS) -c $<

nbody-cuda: nbody-cuda.o wrapper.o update.o calc_acc.o
	$(MPICXX) $(CXXFLAGS)  $^  -o nbody3-cuda $(LDFLAGS)



clean:
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
