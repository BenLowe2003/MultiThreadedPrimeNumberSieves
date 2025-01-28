
#include "MTAtkinsSieve.cu"

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