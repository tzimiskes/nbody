__global__
void update (double * pos, double * vel, double * acc, const int n, double h) {
  size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if( i <  n) {
    for (int k = 0; k < 3; k++) {
      pos[k + 3*i] += vel[k + 3*i]*h + acc[k + 3*i]*h*h/2;
      vel[k + 3*i] += acc[k + 3*i]*h;
    }
  }
}
