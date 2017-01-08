__global__
void calc_acc(double * pos , double * acc, double * mass, const int n, const unsigned int start, const unsigned int end, const unsigned int rank);

__global__
void update (double * pos, double * vel, double * acc, const int n, double h);
