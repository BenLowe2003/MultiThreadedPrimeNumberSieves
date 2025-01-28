

#include "HeaderMain.cuh"




int main() {
    int n = 30;
    int block_size = 256;
    std::cout << "Hello World\n";
    //bool* primes = multithreaded_atkins_sieve(n, block_size);

    /*
    printf("Prime numbers on the GPU:\n");
    for (int i = 0; i < n; ++i) {
        if (primes[i]) {
            std::cout << i << "\n";
        }
    }


    bool* primes_1 = sieveOfAtkin(n);
    bool* primes_2 = sieveOfEratosthenes(n);
    bool* primes_3 = divisibility_testing(n);
    bool* primes_4 = multithreaded_atkins_sieve(n, block_size);


    bool check = true;
    for (int i = 0; i < n; i++) {
        if (primes_1[i] != primes_4[i]) {
            std::cout << i << "\n";
            check = false;
        }
    }

    std::cout << check << "\n";
    */


    int cases[] = { int(pow(2,20)), int(pow(2,21)), int(pow(2,22)), int(pow(2,23)), int(pow(2,24)), int(pow(2,25)), int(pow(2,26)), int(pow(2,26)), int(pow(2,27)), int(pow(2,28))
        ,int(pow(2,29)) };
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



    std::cout << cases[0] << " : " << times[0] << " microseconds\n";
    std::cout << cases[1] << " : " << times[1] << " microseconds\n";
    std::cout << cases[2] << " : " << times[2] << " microseconds\n";





    //sieveOfEratosthenes(50000);
    std::cin.get();



    return 0;

}