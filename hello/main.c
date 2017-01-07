#include <mpi.h>

extern void get_dev_info();

int int main(int argc, char const *argv[]) {
  MPI_Init(&argc, &argv);
  int rank, n_procs;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

  for (int i = 0; i < n_procs; i++) {
    if (rank == i)
      get_dev_info();
  }

  MPI_Finalize();
  return 0;
}
