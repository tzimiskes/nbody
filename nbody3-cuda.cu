#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

#include <aligned_allocator.h>
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
  allocate_device_memory(d_pos, NDIM*n);
  allocate_device_memory(d_vel, NDIM*n);
  allocate_device_memory(d_acc, NDIM*n);
  allocate_device_memory(d_mass, n);
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
  float t_accel = 0, t_update = 0;
  for (int step = 0; step < num_steps; ++step) {
    //cuda events for recording kernel execution time
    // non load balancing partition

    // launch kernel on gpu to calculate acceleration
    t_accel += call_calc_acc(d_pos, d_acc, d_mass, n);

    // launch kernel to update pos and vel
    t_update += call_update(d_pos, d_vel, d_acc, n, dt);

    transfer_from_device(vel, d_vel, NDIM*n);

    // 3. Find the faster moving object.
    if (step % 10 == 0) {
       search(vel, n );
     }
  }

  float nkbytes = (float)((size_t)7 * sizeof(double) * (size_t)n) / 1024.0f;
  //printf("Average time = %f (ms) per step with %d elements %.2f KB over %d steps %.3f%%, %.3f%%, %.3f%%\n", t_calc*1000.0/num_steps, n, nkbytes, num_steps, 100*t_accel/t_calc, 100*t_update/t_calc, 100*t_search/t_calc);
  printf("Average time = %f (ms) per step with %d elements %.2f KB over %d steps %f %f \n", (t_accel+t_update)/num_steps, n, nkbytes, num_steps, t_accel/num_steps, t_update/num_steps);

  Deallocate(pos);
  Deallocate(vel);
  Deallocate(acc);
  Deallocate(mass);

  free_device_memory(d_pos);
  free_device_memory(d_vel);
  free_device_memory(d_acc);
  free_device_memory(d_mass);

  return 0;
}
