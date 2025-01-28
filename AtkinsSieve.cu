

bool* sieveOfAtkin(const int n) {
    bool* numbers = new bool[n];


    for (int i = 0; i <= n; ++i) {
        numbers[i] = false;
    }

    int sqrtN = sqrt(n);

    for (int x = 1; x <= sqrtN; x++) {
        for (int y = 1; y <= sqrtN; y++) {
            int z = 4 * x * x + y * y;
            if (z <= n && (z % 12 == 1 || z % 12 == 5)) {
                numbers[z] = !numbers[z];
            }
            z = 3 * x * x + y * y;
            if (z <= n && z % 12 == 7) {
                numbers[z] = !numbers[z];
            }
            z = 3 * x * x - y * y;
            if (x > y && z <= n && z % 12 == 11) {
                numbers[z] = !numbers[z];
            }
        }
    }

    numbers[3] = numbers[2] = true;

    for (int x = 2; x <= sqrtN; x++) {
        if (numbers[x]) {
            // Corrected loop to mark multiples of primes
            for (int y = x * x; y <= n; y += x) {
                numbers[y] = false;
            }
        }
    }

    return numbers;
}