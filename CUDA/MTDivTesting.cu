

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