#define NDIM (3)
__global__
void update (double * pos, double * vel, double * acc, const int n, double h) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if( i <  n) {
    for (int k = 0; k < NDIM; k++) {
      pos[k + i*NDIM] += vel[k + i*NDIM]*h + acc[k + i*NDIM]*h*h/2;
      vel[k + i*NDIM] += acc[k + i*NDIM]*h;
    }
  }
}
