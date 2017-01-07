NVCC = nvcc
CXX = g++
MPICC = mpicc

CUDA_PATH ?= "/usr/local/cuda-7.5/lib64"
NVCCFLAGS = -ccbin=$(CXX) -lrt -Xcompiler
CXXFLAGS = -O3 -I. -I$(HEADERS) -D_FORCE_INLINES
CFLAGS = -std=c99
LDFLAGS = -lcudart -L/opt/apps/cuda/6.5/lib64/ -L$(CUDA_PATH)

OBJS = hello.o main.o

EXEC = hello

all: $(EXEC)

hello.o: hello.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -c $<

main.o :main.c
	$(MPICC) $(CXXFLAGS) $(CFLAGS) -c $<

$(EXEC): $(OBJS)
	$(MPICC) $(CXXFLAGS)  $^  -o $@ $(LDFLAGS)

clean:
	/bin/rm -fv $(EXEC) *.d *.o *.optrpt
