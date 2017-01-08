#include <stdio.h>

__global__
void calc_acc(double * pos , double * acc, double * mass, const int n,
const unsigned int start, const unsigned int end, const unsigned int rank) {
  // get index of particle
  // need to offset by start to ensure that the index lies in the partiton
  const unsigned int i = threadIdx.x + blockDim.x*blockIdx.x + start;
  double ax = 0, ay = 0, az = 0;

  // make sure we are not acccessing memory that is OOB and
  // is also not out of the partition range
  if (i < end) {
    // this is is ensure MPI is working correctly
    // printf("Rank: %d Thread: %d, local start: %d, local end: %d\n", rank, i, start, end);
    const double xi = pos[0 + 3*i];
    const double yi = pos[1 + 3*i];
    const double zi = pos[2 + 3*i];
    // \vec{a} = \sum\limits_j m[j]*\vec{r'-r}/{|r'-r|^{3/2}|}
    for (int j = 0 ; j < n; ++j) {
      const double dx = pos[0 + 3*j] - xi;
      const double dy = pos[1 + 3*j] - yi;
      const double dz = pos[2 + 3*j] - zi;
      // assume big G is 1
      const double dsq = dx*dx + dy*dy + dz*dz + 1e-50; //padding to prevent divide by zero
      const double m_invR3 = 1.0*mass[j] * rsqrt(dsq)/dsq;

      ax += dx * m_invR3;
      ay += dy * m_invR3;
      az += dz * m_invR3;
    }
    //write acceleration to global mem
    acc[0 + 3*i] = ax;
    acc[1 + 3*i] = ay;
    acc[2 + 3*i] = az;
  }
}
