// This kernel updates position and velocity via improved eulers method
__global__
void update (double * pos, double * vel, double * acc, const int n, double h,
const unsigned int start, const unsigned int end) {
  // see calc_acc.cu for how this works
  const unsigned int i = threadIdx.x + blockDim.x*blockIdx.x + start;
  if(i < end) {
    for (unsigned int k = 0; k < 3; k++) {
      pos[k + 3*i] += vel[k + 3*i]*h + acc[k + 3*i]*h*h/2;
      vel[k + 3*i] += acc[k + 3*i]*h;
    }
  }
}
