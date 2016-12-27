__global__
void calc_acc(double * pos , double * acc, double * mass,  int n);

__global__
void update (double * pos, double * vel, double * acc, const int n, double h);
