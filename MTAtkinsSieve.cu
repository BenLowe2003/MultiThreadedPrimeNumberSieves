#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>


__global__ void initialize(bool* prime, int n, bool value, int y) {
    int x = threadIdx.x;
    if (x < n) {
        prime[y * x] = value;
    }
}


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

bool* multithreaded_atkins_sieve(const int n = 100, int blockSize = 256) {
    size_t size = n * sizeof(bool);
    int threads = static_cast<int>(std::sqrt(n));

    bool* primes = (bool*)malloc(size);

    bool* d_primes;
    cudaMalloc((void**)&d_primes, size);
    cudaMemcpy(d_primes, primes, size, cudaMemcpyHostToDevice);


    int numBlock = (n + blockSize - 1) / blockSize;

    for (int i = 0; i <= threads; i++) {
        initialize <<< 1, threads >>> (d_primes, threads, false, i);
    }

    // this uses n threads but can be optimised for sqrtN threads

  //cudaMemcpy(primes, d_primes, size, cudaMemcpyDeviceToHost);

  /*
  for (int i = 0; i < n; i++) {
      if (primes[i]) {
          std::cout << "error";
      }
  }
  */


    numBlock = (threads + blockSize - 1) / blockSize;

    //First step in Atkins, remove residues
    for (int x = 0; x * x <= n; x++) {

        //std::cout << "x = " << x << "\n" << "n = " << n << "\n" << "threads = " << threads << "\nnumBlock = " << numBlock << "\n\n";

        residue_sieve_ker_1 <<<1, threads >>> (d_primes, x, n);
        residue_sieve_ker_2 <<<1, threads >>> (d_primes, x, n);
        residue_sieve_ker_3 <<<1, threads >>> (d_primes, x, n);
    }

    //cudaMemcpy(primes, d_primes, size, cudaMemcpyDeviceToHost);

    //Print primes without removing squares
    /*
    printf("Prime numbers with prime squares on the GPU:\n");
    for (int i = 0; i < n; ++i) {
        if (primes[i]) {
            std::cout << i << "\n";
        }
    }
    */


    //Second step in Atkins, Remove primes
    for (int x = 2; x * x < n; x++) {

        int threads = static_cast<int>(std::sqrt(n));
        int numBlock = (threads + blockSize - 1) / blockSize;

        if (primes[x]) {
            remove_squares <<<1, threads >>> (d_primes, x, n);
        }
    }


    cudaMemcpy(primes, d_primes, size, cudaMemcpyDeviceToHost);

    /*
    printf("Prime numbers on the GPU:\n");
    for (int i = 0; i < n; ++i) {
        if (primes[i]) {
            std::cout << i << "\n";
        }
    }

    */
    cudaFree(d_primes);

    primes[3] = primes[2] = true;
    primes[1] = false;


    return primes;
}