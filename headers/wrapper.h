void allocate_device_memory(double ** d_pos, double ** d_vel, double ** d_acc, double ** d_mass, int n );
void free_device_memory(double ** d_pos, double ** d_vel, double ** d_acc, double ** d_mass);
void transfer_to_device(double* pos, double* d_pos, int n);
