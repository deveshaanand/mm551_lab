import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load database
data = pd.read_csv("GD2.csv")

# Extract x and y values from the data
xval = data['x'].values
yval = data['y-real'].values

# Plotting the original data
plt.scatter(xval, yval, color='blue')
plt.title('Original Data')
plt.xlabel('xval')
plt.ylabel('yval')
plt.show()

# 2. Create hypothesis function that implements y = mx + c
def hypothesis(xval, m, c):
    return m * xval + c

# 3. Create error function that returns error using MSE equation
def mean_squared_error(yval, Y_cap):
    return np.mean((Y_cap - yval) ** 2)

# 4. Define function for gradient descent with separate learning rates for m and c
def gradient_descent(xval, yval, m, c, alpha_m, alpha_c, iterations):
    error_history = []
    m_history = []
    c_history = []
    
    for i in range(iterations):
        Y_cap = hypothesis(xval, m, c)
        mse = mean_squared_error(yval, Y_cap)
        
        # Update c and m using gradient descent with separate learning rates
        c = c - (2 * alpha_c / len(yval)) * np.sum(Y_cap - yval)
        m = m - (2 * alpha_m / len(yval)) * np.sum((Y_cap - yval) * xval)
        
        # Store the error, m, and c values for each iteration
        error_history.append(mse)
        m_history.append(m)
        c_history.append(c)
        
        # If the error becomes NaN or infinite, stop the iteration
        if np.isnan(mse) or np.isinf(mse):
            break

    return m, c, error_history, m_history, c_history

# 5. Adjust the learning rates for slope and intercept separately for better convergence
learning_rates_m = [0.1, 0.01, 0.001, 0.00005099]
learning_rates_c = [0.1, 0.01,0.025,0.0293,0.029494,0.5,0.6, 0.001, 0.0001,0.00001,0.000001,0.000000000000001]
best_m = 0
best_c = 0
min_error = float('inf')
best_lr_m = 0
best_lr_c = 0
iterations = 1000

for lr_m in learning_rates_m:
    for lr_c in learning_rates_c:
        m, c = 0, 0  # Reset initial m and c for each combination of learning rates
        m, c, error_history, m_history, c_history = gradient_descent(xval, yval, m, c, lr_m, lr_c, iterations)
        
        final_error = error_history[-1]
        
        if final_error < min_error:
            min_error = final_error
            best_m, best_c = m, c
            best_lr_m = lr_m
            best_lr_c = lr_c
        
        print(f"Learning Rate for m: {lr_m}, Learning Rate for c: {lr_c}, Final Error: {final_error}")

# Use the best learning rates to plot the results
m, c, error_history, m_history, c_history = gradient_descent(xval, yval, 0, 0, best_lr_m, best_lr_c, iterations)

# 6. Plot error values vs slope or intercept to see convergence/divergence
plt.figure(figsize=(10,5))

# Plot error vs slope (m)
plt.subplot(1, 2, 1)
plt.plot(m_history, error_history, color='green', label='Error vs Slope (m)')
plt.title('Error vs Slope (m)')
plt.xlabel('Slope (m)')
plt.ylabel('Error (MSE)')
plt.grid(True)

# Plot error vs intercept (c)
plt.subplot(1, 2, 2)
plt.plot(c_history, error_history, color='blue', label='Error vs Intercept (c)')
plt.title('Error vs Intercept (c)')
plt.xlabel('Intercept (c)')
plt.ylabel('Error (MSE)')
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. Plot error values vs iteration to see how error is changing
plt.plot(range(len(error_history)), error_history, color='red')
plt.title('Error vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Error (MSE)')
plt.grid(True)
plt.show()

# 8. Plot the final regression line on the same plot as the original data

# Calculate the predicted y values using the optimized m and c
y_pred = hypothesis(xval, best_m, best_c)

# Plot the original data points
plt.scatter(xval, yval, color='blue', label='Original Data')

# Plot the regression line
plt.plot(xval, y_pred, color='red', label='Fitted Line')

# Add title and labels
plt.title('Original Data with Fitted Regression Line')
plt.xlabel('xval')
plt.ylabel('yval')
plt.legend()
plt.show()

# Print final optimized values
print(f"Optimized Learning Rate for m: {best_lr_m}")
print(f"Optimized Learning Rate for c: {best_lr_c}")
print(f"Optimized m: {best_m}")
print(f"Optimized c: {best_c}")
print(f"Minimum Error: {min_error}")

