__global__
void calc_acc(double * pos , double * acc, double * mass,  const int n , const int start, const int end) {
  // get index of particles
  size_t i = threadIdx.x + blockDim.x * blockIdx.x + start;
  double ax = 0, ay = 0, az = 0;
  // make sure we are not acccessing memory that is OOB
  if (i < end) {
    const double xi = pos[0 + 3*i];
    const double yi = pos[1 + 3*i];
    const double zi = pos[2 + 3*i];

    for (int j = 0 ; j < n; ++j) {
      const double rx = pos[0 + 3*j] - xi;
      const double ry = pos[1 + 3*j] - yi;
      const double rz = pos[2 + 3*j] - zi;
      // assume big G is 1
      const double dsq = rx*rx + ry*ry + rz*rz + 1e-50;
      const double m_invR3 = 1.0*mass[j] * rsqrt(dsq)/dsq;

      ax += rx * m_invR3;
      ay += ry * m_invR3;
      az += rz * m_invR3;
    }
    acc[0 + 3*i] = ax;
    acc[1 + 3*i] = ay;
    acc[2 + 3*i] = az;
  }
}
