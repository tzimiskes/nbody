void allocate_device_memory(double** d_ptr, int n_elements);

void free_device_memory(double** d_ptr);

void transfer_to_device(double* pos, double* d_pos, int n_elements);

void transfer_from_device(double* dst, double* d_src, int n_elements);

float call_calc_acc(double* d_pos, double* d_acc, double* d_mass, const int n,
  const unsigned int start, const unsigned int end, const unsigned int rank);

float call_update(double* d_pos, double* d_vel, double* d_acc, int n, double dt,
  const unsigned int start, const unsigned int end);
