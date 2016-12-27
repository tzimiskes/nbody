#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

#include <cuda.h>
#include <aligned_allocator.h>
#include <cuda_error_check.h>
#include <kernels.cuh>
#include <wrapper.h>

#define NDIM (3)

// Generate a random double between 0,1.
double frand(void) {
  return ((double) rand()) / RAND_MAX;
}

void search (double vel[], const int n)
{
  double minv = 1e10, maxv = 0, ave = 0;
  for (int i = 0; i < n; ++i) {
    double vmag = 0;
    for (int k = 0; k < NDIM; ++k) {

      vmag += (vel[k + i*NDIM] * vel[k + i*NDIM]);
    }
    vmag = sqrt(vmag);
    maxv = fmax(maxv, vmag);
    minv = fmin(minv, vmag);
    ave += vmag;
  }
  printf("min/max/ave velocity = %e, %e, %e\n", minv, maxv, ave / n);
}

void help() {
  fprintf(stderr,"nbody3 --help|-h --nparticles|-n --nsteps|-s --stepsize|-t\n");
}

int main (int argc, char* argv[]) {
  //This is used to check for cuda errors
  // Define the number of particles. The default is 100..
  int n = 100;

  // Define the number of steps to run. The default is 100.
  int num_steps = 100;

  // Pick the timestep size.
  double dt = 0.01;

  for (int i = 1; i < argc; ++i) {
    #define check_index(i,str) \
    if ((i) >= argc) \
    { fprintf(stderr,"Missing 2nd argument for %s\n", str); return 1; }

    if ( strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"--help") == 0)
    {
      help();
      return 1;
    }
    else if (strcmp(argv[i],"--nparticles") == 0 || strcmp(argv[i],"-n") == 0)
    {
      check_index(i+1,"--nparticles|-n");
      i++;
      if (isdigit(*argv[i]))
        n = atoi( argv[i] );
    }
    else if (strcmp(argv[i],"--nsteps") == 0 || strcmp(argv[i],"-s") == 0)
    {
      check_index(i+1,"--nsteps|-s");
      i++;
      if (isdigit(*argv[i]))
        num_steps = atoi( argv[i] );
    }
    else if (strcmp(argv[i],"--stepsize") == 0 || strcmp(argv[i],"-t") == 0)
    {
      check_index(i+1,"--stepsize|-t");
      i++;
      if (isdigit(*argv[i]) || *argv[i] == '.')
        dt = atof( argv[i] );
    }
    else
    {
      fprintf(stderr,"Unknown option %s\n", argv[i]);
      help();
      return 1;
    }
  }
  double *pos  = NULL;
  double *vel  = NULL;
  double *acc  = NULL;
  double *mass = NULL;

  Allocate(pos, n*NDIM);
  Allocate(vel, n*NDIM);
  Allocate(acc, n*NDIM);
  Allocate(mass, n);

  // instantiate pointers
  double * d_pos = NULL;
  double * d_vel = NULL;
  double * d_acc = NULL;
  double * d_mass = NULL;
  // allocate memory on device
  allocate_device_memory(&d_pos, &d_vel, &d_acc , &d_mass, n);
  // Initialize the positions with random numbers (0,1].
  // Particles are given randomized starting positions and slightly
  // varying masses.
  srand(n);
  // copy data from host (CPU) to device (GPU)
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < NDIM; k++) {
      pos[k + i*NDIM] = 2*(frand() - 0.5);
      vel[k + i*NDIM] = 0;
      acc[k + i*NDIM] = 0;
    }
    mass[i] = frand() + DBL_MIN;
  }


  transfer_to_device(d_pos, pos, NDIM*n);
  transfer_to_device(d_vel, vel, NDIM*n);
  transfer_to_device(d_acc, acc, NDIM*n);
  transfer_to_device(d_mass, mass, n);



  // Run the step several times.
  double t_accel = 0, t_update = 0;
  for (int step = 0; step < num_steps; ++step) {
    //cuda events for recording kernel execution time
    cudaEvent_t t1, t2, t3;
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventCreate(&t3);

    cudaEventRecord(t1);
    // non load balancing partition
    dim3 block_size = dim3(256, 1, 1);
    dim3 grid_size = dim3( n/block_size.x + 1, 1, 1);
    // launch kernel on gpu to calculate acceleration
    calc_acc<<<grid_size,block_size>>>(d_pos, d_acc, d_mass, n );
    cudaEventRecord(t2);

    cuda_error_check( cudaGetLastError() );

    // halt cpu execution until the kernel is done
    cudaEventSynchronize(t2);

    // launch kernel to update pos and vel
    update<<<grid_size,block_size>>>( d_pos, d_vel, d_acc, n, dt );


    cuda_error_check( cudaGetLastError() );

    cuda_error_check( cudaMemcpy(vel, d_vel, NDIM*n*sizeof(double), cudaMemcpyDeviceToHost) );

    cudaEventRecord(t3);
    cudaEventSynchronize(t3);


    // 3. Find the faster moving object.
    if (step % 10 == 0) {
       search(vel, n );
     }
    float t_acc, t_up;

    cudaEventElapsedTime(&t_acc, t1, t2);
    cudaEventElapsedTime(&t_up, t2, t3);
    t_accel +=  t_acc;
    t_update += t_up;
  }

  float nkbytes = (float)((size_t)7 * sizeof(double) * (size_t)n) / 1024.0f;
  //printf("Average time = %f (ms) per step with %d elements %.2f KB over %d steps %.3f%%, %.3f%%, %.3f%%\n", t_calc*1000.0/num_steps, n, nkbytes, num_steps, 100*t_accel/t_calc, 100*t_update/t_calc, 100*t_search/t_calc);
  printf("Average time = %f (ms) per step with %d elements %.2f KB over %d steps %f %f \n", (t_accel+t_update)/num_steps, n, nkbytes, num_steps, t_accel/num_steps, t_update/num_steps);

  Deallocate(pos);
  Deallocate(vel);
  Deallocate(acc);
  Deallocate(mass);

  free_device_memory(&d_pos, &d_acc, &d_vel, &d_mass);
  return 0;
}
