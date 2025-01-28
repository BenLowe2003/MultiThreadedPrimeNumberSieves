#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void divisibility_ker(bool* prime, int i, int n) {
    int j = threadIdx.x;
    if ((j > 1) && (j * i < n)) {
        prime[i * j] = false;
    }
}