#include <stdio.h>
#include <cuda.h>
#include <mpi.h>
int main() {

  MPI_Init(&argc,&argv);
  // get processor rank and number of processors
  int rank, n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for(int j = 0; j < n_procs; ++j) {
    if (rank == j) {
      int nDevices;
      printf("Hello from worker %d of %d\n", rank , n_procs);
      cudaGetDeviceCount(&nDevices);
      for (int i = 0; i < nDevices; i++) {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
        prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
        prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
        2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
      }
    }
  }
}
