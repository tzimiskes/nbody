#define NDIM (3)

__global__
void calc_acc(double * pos , double * acc, double * mass,  int n) {
  // get index of particles
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  double ax = 0, ay = 0, az = 0;
  const double xi = pos[0 + i*NDIM];
  const double yi = pos[1 + i*NDIM];
  const double zi = pos[2 + i*NDIM];
  // make sure we are not acccessing memory that is OOB
  if (i < n) {
    for (int j = 0 ; j < n; ++j) {


      double rx = pos[0 + j*NDIM] - xi;
      double ry = pos[1 + j*NDIM] - yi;
      double rz = pos[2 + j*NDIM] - zi;
      double dsq = rx*rx + ry*ry + rz*rz + 1e-50;
      double m_invR3 = mass[j]*rsqrt(dsq) / dsq;

      ax += rx * m_invR3;
      ay += ry * m_invR3;
      az += rz * m_invR3;
    }
    acc[0 + i*NDIM] = 1.0 * ax;
    acc[1 + i*NDIM] = 1.0 * ay;
    acc[2 + i*NDIM] = 1.0 * az;
  }
}
