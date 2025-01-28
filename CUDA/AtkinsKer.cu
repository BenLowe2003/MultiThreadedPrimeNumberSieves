#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>


__global__ void residue_sieve_ker_1(bool* prime, int x, int n) {
    int y = threadIdx.x;
    if (y * y <= n) {
        int z = 4 * x * x + y * y;
        if ((z <= n) && ((z % 12 == 1) || (z % 12 == 5))) {
            prime[z] = !prime[z];
        }
    }
}

__global__ void residue_sieve_ker_2(bool* prime, int x, int n) {
    int y = threadIdx.x;
    if (y * y <= n) {
        int z = 3 * x * x + y * y;
        if ((z <= n) && (z % 12 == 7)) {
            prime[z] = !prime[z];
        }
    }
}

__global__ void residue_sieve_ker_3(bool* prime, int x, int n) {
    int y = threadIdx.x;
    if (y * y <= n) {
        int z = 3 * x * x - y * y;
        if ((z <= n) && ((z % 12 == 11) && (x > y))) {
            prime[z] = !prime[z];
        }
    }
}



__global__ void remove_squares(bool* prime, int x, int n) {
    int y = threadIdx.x;
    if ((y * y < n) && ((y != 0) || (y != 1))) {
        int x_squared = x * x;
        int remove_square = x_squared * y;
        prime[remove_square] = false;
    }
}