# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 08:37:13 2024

@author: Devesh Anand
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 10 # Lattice size (M x M)
steps = 1000 # Number of steps for the random walk
trials = 500  # Number of trials to average over

def random_walk(steps, M):
    # Starting position at the center of the lattice
    x, y = M//2, M//2
   
    # Record the squared displacement
    squared_displacement = np.zeros(steps)
   
    for step in range(steps):
        # Choose a random direction: up, down, left, right
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up':
            x -= 1
        elif direction == 'down':
            x += 1
        elif direction == 'left':
            y -= 1
        elif direction == 'right':
            y += 1
       
        # Ensure the walker stays within the lattice boundaries
        x = np.clip(x, 0, M-1)
        y = np.clip(y, 0, M-1)
       
        # Calculate squared displacement from the origin
        squared_displacement[step] = x**2 + y**2
   
    return squared_displacement

# Perform multiple trials and average the squared displacement
mean_squared_displacement = np.zeros(steps)

for _ in range(trials):
    squared_displacement = random_walk(steps, M)
    mean_squared_displacement += squared_displacement

mean_squared_displacement /= trials

# Plotting the mean-square displacement
plt.figure(figsize=(10, 6))
plt.plot(np.arange(steps), mean_squared_displacement, label='Mean-Square Displacement')
plt.xlabel('Number of Steps')
plt.ylabel('Mean-Square Displacement')
plt.title('Mean-Square Displacement vs Number of Steps ')
plt.grid(True)
plt.legend()
plt.show()