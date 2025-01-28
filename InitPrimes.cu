#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void initialize(bool* prime, int n, bool value, int y) {
    int x = threadIdx.x;
    if (x < n) {
        prime[y * x] = value;
    }
}