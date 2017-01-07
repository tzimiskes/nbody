#include <stdio.h>
#include <cuda.h>

extern "C" void get_dev_info(int rank, int n_procs) {

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  printf("Hello from %d of %d\n", rank+1, n_procs);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("DeviceNumber: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
  cudaDeviceSynchronize();
}
