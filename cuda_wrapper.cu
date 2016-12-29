#include <kernels.cuh>

#include <stdlib.h>
#include <stdio.h>

// macro to handle CUDA errors
#define cuda_error_check( _cmd_ ) \
{ \
   /* Execute the function and capture the return int. */ \
   cudaError_t cuda_error = (_cmd_); \
   /* Check the error flag. Print if something went wrong and then abort. */ \
   if (cuda_error != cudaSuccess) \
   { \
     fprintf(stderr, "Cuda runtime error! %s  \n Error code: %i -- %s!\n",#_cmd_, cuda_error, cudaGetErrorString(cuda_error)); \
     exit(EXIT_FAILURE); \
   } \
}

//wrappers for various cuda functions
//provides a layer of abstraction so main can be compiled mpicxx

void allocate_device_memory(double* &d_ptr, int n_elements) {
  cuda_error_check( cudaMalloc((void **)&d_ptr, n_elements*sizeof(double)) );
}

void free_device_memory(double* &d_ptr)  {
  cuda_error_check(cudaFree(d_ptr));
  d_ptr = NULL;
}

void transfer_to_device(double* d_dst, double* src, int n_elements) {
  cuda_error_check( cudaMemcpy(d_dst, src, n_elements*sizeof(double), cudaMemcpyHostToDevice) );
}

void transfer_from_device(double* dst, double* d_src, int n_elements) {
  cuda_error_check( cudaMemcpy(dst, d_src, n_elements*sizeof(double), cudaMemcpyDeviceToHost) );
}


float call_calc_acc(double* d_pos, double* d_acc, double* d_mass, int n) {
  cudaEvent_t t0, t1;
  cuda_error_check( cudaEventCreate(&t0) );
  cuda_error_check( cudaEventCreate(&t1) );

  dim3 block_size = dim3(256, 1, 1);
  dim3 grid_size = dim3( n/block_size.x + 1, 1, 1);

  cuda_error_check( cudaEventRecord(t0) );
  calc_acc<<<grid_size, block_size>>>(d_pos, d_acc, d_mass, n);
  cuda_error_check( cudaEventRecord(t1));

  cudaDeviceSynchronize();

  float t_ellapsed;
  cudaEventElapsedTime(&t_ellapsed, t0, t1);

  return t_ellapsed;
}

float call_update(double* d_pos, double* d_vel,double* d_acc, int n, double dt) {
  cudaEvent_t t0, t1;
  cuda_error_check( cudaEventCreate(&t0));
  cuda_error_check( cudaEventCreate(&t1));

  dim3 block_size = dim3(256, 1, 1);
  dim3 grid_size = dim3( n/block_size.x + 1, 1, 1);

  cuda_error_check( cudaEventRecord(t0) );
  update<<<grid_size,block_size>>>( d_pos, d_vel, d_acc , n, dt);
  cuda_error_check( cudaEventRecord(t1) );

  cudaDeviceSynchronize();

  float t_ellapsed;
  cudaEventElapsedTime(&t_ellapsed, t0, t1);

  return t_ellapsed;
}
