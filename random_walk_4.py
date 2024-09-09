import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 10  # Number of particles
M = 50 # Size of the lattice (MxM)
num_steps = 10 # Number of steps

# Energy function
def energy(i, j, M):
    return -(i**2 + j**2) / (2 * M**2)

# Initialize lattice with N particles
lattice = np.zeros((M, M))
particle_positions = []
while len(particle_positions) < N:
    x, y = np.random.randint(0, M, size=2)
    if lattice[x, y] == 0:
        lattice[x, y] = 1
        particle_positions.append((x, y))

displacement = []
total_energy = []

# Perform the biased random walk for all particles simultaneously
for _ in range(num_steps):
    new_positions = []
    for x, y in particle_positions:
        direction = np.random.choice(['up', 'down', 'left', 'right'])
        
        if direction == 'up':
            new_y = (y + 1) % M
            new_x = x
        elif direction == 'down':
            new_y = (y - 1) % M
            new_x = x
        elif direction == 'left':
            new_x = (x - 1) % M
            new_y = y
        elif direction == 'right':
            new_x = (x + 1) % M
            new_y = y
        
        delta_E = energy(new_x, new_y, M) - energy(x, y, M)
        Pmove = np.exp(delta_E)
        
        if lattice[new_x, new_y] == 0 and np.random.rand() < Pmove:  # Move based on probability
            lattice[x, y] = 0
            lattice[new_x, new_y] = 1
            new_positions.append((new_x, new_y))
        else:
            new_positions.append((x, y))
    
    particle_positions = new_positions
    displacement.append(np.sum([x**2 + y**2 for x, y in particle_positions]))
    total_energy.append(np.sum([energy(x, y, M) for x, y in particle_positions]))

# Calculate mean square displacement
msd = np.cumsum(displacement) / np.arange(1, num_steps + 1)

# Plot mean square displacement and total energy
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(msd)
plt.xlabel('Number of Steps')
plt.ylabel('Mean Square Displacement')
plt.title('Biased Random Walk (Mean Square Displacement)')

plt.subplot(1, 2, 2)
plt.plot(total_energy)
plt.xlabel('Number of Steps')
plt.ylabel('Total Energy')
plt.title('Biased Random Walk (Total Energy)')

plt.tight_layout()
plt.show()
