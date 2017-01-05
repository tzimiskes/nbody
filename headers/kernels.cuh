__global__
void calc_acc(double * pos , double * acc, double * mass, const int n, const int start, const int end);

__global__
void update (double * pos, double * vel, double * acc, const int n, double h);
