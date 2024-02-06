#!/usr/bin/env python3
import math 

class Poisson:
    """
    Represents a Poisson distribution.
    
    Attributes:
        lambtha: The expected number of occurrences in a given time frame.
    """


class Poisson:
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        fact = 1
        for i in range(1, n + 1):
            fact *= i
        return fact

    def exp(self, x):
        n, term, sum = 0, 1, 1
        while abs(term) > 1e-10:
            n += 1
            term *= x / n
            sum += term
        return sum

    def pmf(self, k):
        if k < 0:
            return 0
        k = int(k)
        e = self.exp(-self.lambtha)
        return e * self.lambtha**k / self.factorial(k)