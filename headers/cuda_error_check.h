#ifndef __my_cuda_header_h
#define __my_cuda_header_h

#include <stdio.h>

#define cuda_error_check( _cmd_ ) \
{ \
   /* Execute the function and capture the return int. */ \
   cudaError_t cuda_error = (_cmd_); \
   /* Check the error flag. Print if something went wrong and then abort. */ \
   if (cuda_error != cudaSuccess) \
   { \
     fprintf(stderr, "Cuda runtime error! %s  \n Error code: %i -- %s!\n",#_cmd_, cuda_error, cudaGetErrorString(cuda_error)); \
     exit(EXIT_FAILURE); \
   } \
}

#endif
