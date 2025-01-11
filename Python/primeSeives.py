import numpy as np

def eratosthenes(n):
    primes = [True for i in range(n)]
    for i in range(2,n):
        if primes[i] == True:
            for j in range(2,int(np.sqrt(n))):
                if i*j < n:
                    primes[i*j] = False
    return [i for i in range(n) if primes[i] == True]



def divisibility_testing(n):
    primes = []
    for i in range(n):
        is_prime = True
        for j in range(2,int(np.sqrt(n))):
            if i % j == 0:
                is_prime = False
        if is_prime:
            primes.append(i)
    return primes

def segmented_eratosthenes(n):
    segment_width = int(np.sqrt(n))
    segment_num = np.ceil(n/np.sqrt(n))
    first_primes = eratosthenes(segment_width)

    #for segment in range(1, segment_num):
        #segment = [True for ]

def bens_seive(n):
    numbers = [True for i in range(2,n)]
    for i in range(0,n-2):
        if numbers[i] == True:
            primes = [x+2 for x in range(0,i) if (numbers[x] == True) & (x >= 0)]
            for j in primes:
                if i*j < (n-2):
                    numbers[i*j] = False
    return [i+2 for i in range(n-2) if numbers[i] == True]

def atkins(n):
    #Initialise List
    primes = [False] * n

    #Switch integars that satisfy the binary Quadratic forms generated in Atkins papers by the delta set
    for x in range(1, int(np.sqrt(n)) + 1):
        for y in range(1, int(np.sqrt(n)) + 1):

            z = 4 * x**2 + y**2
            if ((z % 12 == 1) or (z % 12 == 5)) and (z <= n):
                primes[z] = not primes[z]

            z = 3*x**2 + y**2
            if (z % 12 == 7) and (z <= n):
                primes[z] = not primes[z]

            z = 3*x**2 - y**2
            if (z % 12 == 11) and (z <= n) and (x > y):
                primes[z] = not primes[z]
        
    #Remove Squares
    for x in range(5,int(np.sqrt(n))+1):
        if primes[x]:
            for y in range(x**2, n, x**2):
                primes[y] = False
    
    #Print Primes
    return [i for i in range(n) if primes[i] == True]



