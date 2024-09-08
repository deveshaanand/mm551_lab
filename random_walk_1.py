# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 08:37:01 2024

@author: Devesh Anand
"""
import numpy as np
import matplotlib.pyplot as plt

# Function to perform a random walk and return displacement and mean-squared displacement
def random_walk(num_steps, num_trials):
    displacement = np.zeros(num_steps)
    mean_squared_displacement = np.zeros(num_steps)

    for _ in range(num_trials):
        position = np.zeros(num_steps)
        for t in range(1, num_steps):
            step = np.random.choice([-1, 1])
            position[t] = position[t-1] + step
        displacement += position
        mean_squared_displacement += position**2

    displacement /= num_trials
    mean_squared_displacement /= num_trials
   
    return displacement, mean_squared_displacement

# Parameters
num_steps = 50  # Number of time steps

# Perform random walks for different numbers of trials
trials = [10, 100, 1000]
results = {}

for num_trials in trials:
    print(f"Performing {num_trials} trials...")
    displacement, mean_squared_displacement = random_walk(num_steps, num_trials)
    results[num_trials] = mean_squared_displacement

# Plotting
plt.figure(figsize=(14, 6))

# Plot mean-squared displacement for different trial numbers
for num_trials in trials:
    plt.plot(results[num_trials], label=f'{num_trials} Trials')

plt.xlabel('Number of Steps')
plt.ylabel('MSD')
plt.title('Mean-Squared Displacement vs Time')
plt.legend()
plt.grid(True)
plt.show()