

bool* divisibility_testing(const int n) {
    bool* numbers = new bool[n];

    for (int i = 0; i < n; ++i) {
        numbers[i] = true;
    }

    for (int i = 2; i < n; i++) {
        for (int j = 2; i * j <= n; j++) {
            numbers[i * j] = false;
        }
    }

    numbers[0] = numbers[1] = false;

    return numbers;
}