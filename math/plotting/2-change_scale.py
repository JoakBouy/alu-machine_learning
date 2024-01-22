#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

# your code here
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.yscale('log')
plt.suptitle('Exponential Decay of c-14')
plt.plot(x, y, '-')

plt.show()