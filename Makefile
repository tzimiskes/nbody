HEADERS	= ./headers
KERNELS = ./kernels
NVCC ?= nvcc
CXX = clang++

CUDA_PATH ?= "/usr/local/cuda-8.0"
NVCCFLAGS = -ccbin=$(CXX) -Xcompiler
CXXFLAGS = -O3 -I. -I$(HEADERS) -D_FORCE_INLINES

EXEC = nbody3-cuda

all: $(EXEC)



# Load common make options
LDFLAGS	=  -L/opt/apps/cuda/6.5/lib64/lcudart -L$(CUDA_HOME)/lib64

update.o: $(KERNELS)/update.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
calc_acc.o: $(KERNELS)/calc_acc.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
wrapper.o: wrapper.cu
		$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<

nbody3-cuda.o :nbody3-cuda.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<

nbody3-cuda: update.o calc_acc.o  wrapper.o nbody3-cuda.o
	$(NVCC) $(CXXFLAGS) $(NVFLAGS)  $^ -o nbody3-cuda

clean:
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
