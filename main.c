#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <string.h>

#include <mpi.h>

#include <cuda_wrapper.h>

// Generate a random double between 0,1.
double frand(void) {
  return ((double) rand()) / RAND_MAX;
}
// orints the current min max ave velocity. Used mostly for diagnostics
void search (const double vel[], const int n)
{
  double minv = 1e10, maxv = 0, ave = 0;
  for (size_t i = 0; i < n; ++i) {
    double vmag = 0;
    for (int k = 0; k < 3; ++k) {

      vmag += (vel[k + 3*i] * vel[k + 3*i]);
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

void partition_range (const int global_start, const int global_end,
                     const int num_partitions, const int rank,
                     int* local_start, int* local_end)
{
  // Total length of the iteration space.
  const int global_length = global_end - global_start;

  // Simple per-partition size ignoring remainder.
  const int chunk_size = global_length / num_partitions;

  // And now the remainder.
  const int remainder = global_length - chunk_size * num_partitions;

  // We want to spreader the remainder around evening to the 1st few ranks.
  // ... add one to the simple chunk size for all ranks < remainder.
  if (rank < remainder)
  {
    *local_start = global_start + rank * chunk_size + rank;
    *local_end   = *local_start + chunk_size + 1;
  }
  else
  {
    *local_start = global_start + rank * chunk_size + remainder;
    *local_end   = *local_start + chunk_size;
  }
}

int main (int argc, char* argv[]) {
  // Define the number of particles. The default is 100..
  int n = 100;
  // Initialize MPI
  MPI_Init(&argc, &argv);
  // get processor rank and number of processors
  int rank, n_procs;
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Pick the timestep size.
  double h = 0.01;
  // Define the number of steps to run. The default is 100.
  int num_steps = 100;

  // parse args
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
        h = atof( argv[i] );
    }
    else
    {
      fprintf(stderr,"Unknown option %s\n", argv[i]);
      help();
      return 1;
    }
  }
  // allocate mem for host pointers
  double* pos  = (double* )malloc(n*3*sizeof(double));
  double* vel  = (double* )malloc(n*3*sizeof(double));
  double* acc  = (double* )malloc(n*3*sizeof(double));
  double* mass = (double* )malloc(n*sizeof(double));

  // instantiate device pointers
  double* d_pos  = NULL;
  double* d_vel  = NULL;
  double* d_acc  = NULL;
  double* d_mass = NULL;
  // allocate memory on device
  allocate_device_memory(&d_pos,  n*3);
  allocate_device_memory(&d_vel,  n*3);
  allocate_device_memory(&d_acc,  n*3);
  allocate_device_memory(&d_mass, n);

  // Initialize the positions with random numbers (0,1].
  // Particles are given randomized starting positions and slightly
  // varying masses.
  // have rank 0 set initial values
  if (rank == 0) {
    srand(n);

    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < 3; k++) {
        pos[k + i*3] = 2*(frand() - 0.5);
        vel[k + i*3] = 0;
        acc[k + i*3] = 0;
      }
      mass[i] = frand() + DBL_MIN;
    }
  }
  // broadcast data to other processors
  MPI_Bcast(pos, 3*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(vel, 3*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(acc, 3*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(mass,  n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // copy data from host (CPU) to device (GPU)
  transfer_to_device(d_pos, pos, 3*n);
  transfer_to_device(d_vel, vel, 3*n);
  transfer_to_device(d_acc, acc, 3*n);
  transfer_to_device(d_mass, mass, n);

  int local_start, local_end;
  partition_range(0, n, n_procs, rank, &local_start, &local_end);

  /////////////// Copied from stack overflow.
  int counts [n_procs];
  int disps  [n_procs];

  for (int i = 0; i < n_procs; ++i) {
    int temp_start, temp_stop;
    partition_range(0, n, n_procs, i, &temp_start, &temp_stop);
    counts[i] = 3*(temp_stop - temp_start);
  }

  disps[0] = 0;
  for (int i = 1; i < n_procs; ++i)
    disps[i] = disps[i-1] + counts[i-1];

  // keep track of how long it takes to run these functions
  float t_accel = 0, t_update = 0;
  for (int step = 0; step < num_steps; ++step) {

    // launch kernel on gpu to calculate acceleration
    // retuns time it takes to complete function
    t_accel += call_calc_acc(d_pos, d_acc, d_mass, n, local_start, local_end, rank);

    // launch kernel to update pos and vel
    t_update += call_update(d_pos, d_vel, d_acc, n, h, local_start, local_end);

    transfer_from_device(pos, d_pos, 3*n);
    transfer_from_device(vel, d_vel, 3*n);

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pos, counts, disps, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, vel, counts, disps, MPI_DOUBLE, MPI_COMM_WORLD);

    transfer_to_device(d_pos, pos, 3*n);
    transfer_to_device(d_vel, vel, 3*n);
        // 3. Find the faster moving object.
    if(rank == 0) {
      if (step % 10 == 0) {
        search(vel, n );
      }
    }
  }


  float nkbytes = (float)((size_t)7 * sizeof(double) * (size_t)n) / 1024.0f;
  if (rank == 0)
  printf("Average time = %f (ms) per step with %d elements %.2f KB over %d steps %f %f \n",
    (t_accel+t_update)/num_steps, n, nkbytes, num_steps, t_accel/num_steps, t_update/num_steps);

  free(pos);
  pos = NULL;
  free(vel);
  vel = NULL;
  free(acc);
  acc = NULL;
  free(mass);
  mass = NULL;

  free_device_memory(&d_pos);
  free_device_memory(&d_vel);
  free_device_memory(&d_acc);
  free_device_memory(&d_mass);

  MPI_Finalize();

  return 0;
}
