HEADERS	= ./headers
KERNELS = ./kernels

NVCC = nvcc
MPICXX = mpicxx

CUDA_PATH ?= "/usr/local/cuda-8.0"

NVCCFLAGS = -ccbin=$(CXX) -Xcompiler
CXXFLAGS = -O3 -I. -I$(HEADERS) -D_FORCE_INLINES
LDFLAGS = -lcudart -L/opt/apps/cuda/6.5/lib64/lcudart -L/usr/local/cuda-8.0/lib64/

EXEC = nbody-cuda

all: $(EXEC)

update.o: $(KERNELS)/update.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
calc_acc.o: $(KERNELS)/calc_acc.cu
	$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<
wrapper.o: wrapper.cu
		$(NVCC) $(CXXFLAGS) $(NVFLAGS) -c $<

main.o :main.cpp
	$(MPICXX) $(CXXFLAGS) $(NVFLAGS) -c $<

$(EXEC): main.o wrapper.o update.o calc_acc.o
	$(MPICXX) $(CXXFLAGS)  $^  -o $@ $(LDFLAGS)

clean:
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
