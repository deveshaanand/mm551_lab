# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 08:37:45 2024

@author: Devesh Anand
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
M = 50 # Lattice size (M x M)
N = 10 # Number of particles
steps = 1000  # Number of steps for the random walk
trials = 100  # Number of trials to average over

# Function to perform a single random walk
def random_walk(M, N, steps):
    # Initialize the lattice with zeros
    lattice = np.zeros((M, M), dtype=int)

    # Place N-1 stationary particles randomly on the lattice
    stationary_positions = np.random.choice(M*M, N-1, replace=False)
    for pos in stationary_positions:
        x, y = divmod(pos, M)
        lattice[x, y] = 1
   
    # Start the random walk with the last particle
    pos = np.random.choice(np.setdiff1d(np.arange(M*M), stationary_positions))
    x, y = divmod(pos, M)
   
    # Record initial position
    initial_position = np.array([x, y])
    displacement = np.zeros(steps)
   
    for step in range(steps):
        # Choose a random direction: up, down, left, right
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up' and x > 0:
            x -= 1
        elif direction == 'down' and x < M-1:
            x += 1
        elif direction == 'left' and y > 0:
            y -= 1
        elif direction == 'right' and y < M-1:
            y += 1

        # Avoid moving into an occupied position
        if lattice[x, y] == 1:
            if direction == 'up':
                x += 1
            elif direction == 'down':
                x -= 1
            elif direction == 'left':
                y += 1
            elif direction == 'right':
                y -= 1
       
        # Calculate displacement
        displacement[step] = np.linalg.norm(np.array([x, y]) - initial_position)
   
    return displacement

# Perform multiple trials and calculate the mean square displacement
mean_square_displacement = np.zeros(steps)

for _ in range(trials):
    displacement = random_walk(M, N, steps)
    mean_square_displacement += displacement**2  # Square the displacement

mean_square_displacement /= trials  # Average over trials

# Plotting the mean square displacement
plt.figure(figsize=(10, 6))
plt.plot(np.arange(steps), mean_square_displacement, label='Mean Square Displacement')
plt.xlabel('Number of Steps')
plt.ylabel('Mean Square Displacement')
plt.title('Mean Square Displacement vs Number of Steps for a Random Walk')
plt.grid(True)
plt.legend()
plt.show()