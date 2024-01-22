#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# your code here
names = ['Buoy', 'Gai', 'Joak']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
orange = (1, 0.5, 0)
peach = (1, 0.89, 0.71)
fruit_colors = ['red', 'yellow', orange, peach]

fig, ax = plt.subplots(figsize=(8, 6))

bottom = np.zeros(len(names))

bar_width = 0.5

for i, fruit_name in enumerate(fruits):
    plt.bar(
        names,
        fruit[i],
        bar_width,
        label=fruit_name,
        color=fruit_colors[i],
        bottom=bottom
    )

    # Update the bottom positions for the next fruit
    bottom += fruit[i]

plt.ylim(0, 80)
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')

plt.legend(fruits)
plt.show()