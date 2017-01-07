NVCC = nvcc
CXX = g++
MPICXX = mpicc

CUDA_PATH ?= "/usr/local/cuda-7.5/lib64"
NVCCFLAGS = -ccbin=$(CXX) -lrt 
CXXFLAGS = -O3 -I. -I$(HEADERS) -D_FORCE_INLINES -std=c99
LDFLAGS = -lcudart -L/opt/apps/cuda/6.5/lib64/ -L$(CUDA_PATH) -lm

OBJS = hello.o main.o

EXEC = hello

all: $(EXEC)

hello.o: hello.cu
	$(NVCC) $(NVCCFLAGS)

main.o :main.c
	$(MPICXX) $(CXXFLAGS) -c $<

$(EXEC): $(OBJS)
	$(MPICXX) $(CXXFLAGS)  $^  -o $@ $(LDFLAGS)

clean:
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
