#include<cuda_error_check.h>
#include<cuda_runtime.h>

#include <stdlib.h>
#include <stdio.h>

#define NDIM (3)
void allocate_device_memory( double ** d_pos, double ** d_acc, double ** d_vel, double ** d_mass , int n ) {

  double * temp = NULL;
  cuda_error_check( cudaMalloc( (void **)&temp, NDIM*n*sizeof(double) ));
  *d_pos = temp;

  temp = NULL;
  cuda_error_check( cudaMalloc( (void **)&temp, NDIM*n*sizeof(double) ));
  *d_vel = temp;

  temp = NULL;
  cuda_error_check( cudaMalloc( (void **)&temp, NDIM*n*sizeof(double) ));
  *d_acc = temp;

  temp = NULL;
  cuda_error_check( cudaMalloc( (void **)&temp, n*sizeof(double) ));
  *d_mass = temp;
}

void free_device_memory(double ** d_pos, double ** d_acc, double ** d_vel, double ** d_mass)  {
  cudaFree(*d_pos);
  *d_pos = NULL;
  cudaFree(*d_vel);
  *d_vel = NULL;
  cudaFree(*d_acc);
  *d_acc = NULL;
  cudaFree(*d_mass);
  *d_mass = NULL;
  cuda_error_check ( cudaGetLastError() );
}

void transfer_to_device(double* dst, double* src, int n) {
  cuda_error_check( cudaMemcpy(dst, src, n*sizeof(double), cudaMemcpyHostToDevice) );
}
