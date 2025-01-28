

bool* sieveOfEratosthenes(const int n) {
    bool* numbers = new bool[n];

    for (int i = 0; i < n; ++i) {
        numbers[i] = true;
    }

    numbers[0] = numbers[1] = false;

    for (int i = 2; i * i < n; i++) {
        if (numbers[i]) {
            for (int j = i * i; j < n; j += i) {
                numbers[j] = false;
            }
        }
    }

    return numbers;
}