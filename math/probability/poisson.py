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

    def pmf(self, k):
        if k < 0:
            return 0
        k = int(k)
        e = math.exp(-self.lambtha)
        return e * self.lambtha**k / math.factorial(k)