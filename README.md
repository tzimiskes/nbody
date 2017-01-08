# nbody
CUDA  and MPI accelerated nbody problem.

Now with 100% more MPI!

Tested on the Stampede down in Texas and using 4 GPUS to do 80000 particles is 4x as fast as using 1 (who woulda thunk?).

In the future, I might look to improve this project by looking at better CUDA implementations (IE using shared memory), a distributed memory implementation for MPI, and general code optimizations (I'm sure I can optimize memory access more).
