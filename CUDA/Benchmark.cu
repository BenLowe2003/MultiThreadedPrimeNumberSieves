#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

#include <chrono>
#include <stdio.h>

#include "SingleSieves.cu"


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

__global__ void initialize(bool* prime, int n, bool value, int y) {
    int x = threadIdx.x;
    if (x < n) {
        prime[y * x] = value;
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
        initialize << < 1, threads >> > (d_primes, threads, false, i);
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

        residue_sieve_ker_1 << <1, threads >> > (d_primes, x, n);
        residue_sieve_ker_2 << <1, threads >> > (d_primes, x, n);
        residue_sieve_ker_3 << <1, threads >> > (d_primes, x, n);
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
            remove_squares << <1, threads >> > (d_primes, x, n);
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


__global__ void divisibility_ker(bool* prime, int i, int n) {
    int j = threadIdx.x;
    if ((j > 1) && (j * i < n)) {
        prime[i * j] = false;
    }
}

bool* multithreaded_divisibility_testing(const int n = 100, int blockSize = 256) {

    size_t size = n * sizeof(bool);
    bool* primes = (bool*)malloc(size);

    bool* d_primes;
    cudaMalloc((void**)&d_primes, size);
    cudaMemcpy(d_primes, primes, size, cudaMemcpyHostToDevice);


    int numBlock = (n + blockSize - 1) / blockSize;

    initialize << < 1, n >> > (d_primes, n, true, 1); // this uses n threads but can be optimised for sqrtN threads 

    int threads = n / 2;

    //numBlock = (threads + blockSize - 1) / blockSize;

    int sqrtN = sqrt(n);


    for (int i = 2; i < sqrtN; i++) {

        divisibility_ker << < 1, sqrtN >> > (d_primes, i, n);

    }

    cudaMemcpy(primes, d_primes, size, cudaMemcpyDeviceToHost);

    cudaFree(d_primes);

    return primes;
}




int print(bool* numbers, int n) {
    for (int i = 2; i < n; i++) {
        if (numbers[i]) {
            std::cout << i << "\n";
        }
    }

    return 0;
}

double* time_it(int* cases, int num_cases) {
    double* times = new double[num_cases];


    for (int i = 0; i < num_cases; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        multithreaded_atkins_sieve(cases[i]);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times[i] = duration.count();
    }

    return times;
}

int main() {
    int n = 30;
    int block_size = 256;
    std::cout << sieveOfAtkin(10);



    int cases[] = { pow(2,20), pow(2,21), pow(2,22), pow(2,23), pow(2,24), pow(2,25), pow(2,26), pow(2,27), pow(2,28), pow(2,29)
        , pow(2,30), /*pow(2,31), pow(2,32), pow(2,34), pow(2,36), pow(2,38), pow(2,40)*/ };
    int num_cases = sizeof(cases) / sizeof(cases[0]);
    double* times = time_it(cases, num_cases);

    std::cout << "cases = [";
    for (int i = 0; i < num_cases; i++) {
        std::cout << " " << cases[i] << ",";
    }
    std::cout << "] \n";

    std::cout << "times = [";
    for (int i = 0; i < num_cases; i++) {
        std::cout << " " << times[i] << ",";
    }
    std::cout << "] \n";




    std::cout << "fin";



    return 0;

}