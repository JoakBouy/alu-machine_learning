#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# Create a custom layout
plt.figure(figsize=(15, 8))

# Set the font size to 'x-small' for all labels and titles
cs = 7.0
plt.xlabel('X-axis label', fontsize=cs)

# First graph
plt.subplot(3, 2, 1)
x0 = np.arange(0, 11)
plt.xlim(0, 10)
plt.ylim(0, 1010)
plt.yticks(np.arange(0, 1010, 500))
plt.plot(x0, y0, '-r')

# second graph
plt.subplot(3, 2, 2)
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.scatter(x1, y1, color='magenta')
plt.xticks(np.arange(60, 90, 10))
plt.yticks(np.arange(170, 193, 10))
plt.title("Men's Height vs Weight")

# third graph
plt.subplot(3, 2, 3)
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.yscale('log')
plt.suptitle('Exponential Decay of c-14')
plt.xticks(np.arange(0, 28650, 10000))
plt.xlim(0, 28650)
plt.plot(x2, y2, '-')

# fourth graph
plt.subplot(3, 2, 4)
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.subplots_adjust(top=0.9)
plt.xlim(0, 20000)
plt.xticks(np.arange(0, 20010, 5000))
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.5))
plt.plot(x3, y31, '--r', label='C-14')
plt.plot(x3, y32, '-g', label='Ra-226')
plt.legend(loc='upper right')

# fifth graph occupying the whole space of the bottom row
plt.subplot(3, 2, 5)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.yticks(range(0, 31, 5))
plt.xticks(range(0, 101, 10))
plt.xlim(0, 100)
plt.ylim(0, 30)
plt.hist(student_grades, bins=range(0, 101, 10),
         edgecolor='k', alpha=0.5, color='green')

plt.tight_layout()
plt.suptitle('All in One')
plt.show()