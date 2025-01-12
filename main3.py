import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# File paths for data
x_data_path = r'C:\Users\KIIT\Desktop\ML\linearX.csv'
y_data_path = r'C:\Users\KIIT\Desktop\ML\linearY.csv'

# Load data
x = pd.read_csv(x_data_path, header=None).values
y = pd.read_csv(y_data_path, header=None).values

# Feature normalization
x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
x = np.c_[np.ones(x.shape[0]), x]  

def batch_gradient_descent(x, y, lr, max_iter):
    m = len(y)
    theta = np.zeros(x.shape[1])
    cost_history = []
    
    for _ in range(max_iter):
        predictions = np.dot(x, theta)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        cost_history.append(cost)
        
        gradient = (1 / m) * np.dot(x.T, predictions - y)
        theta -= lr * gradient
        
    return theta, cost_history

# Define function for Stochastic Gradient Descent
def stochastic_gradient_descent(x, y, lr, max_iter):
    m = len(y)
    theta = np.zeros(x.shape[1])
    cost_history = []
    
    for _ in range(max_iter):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = x[random_index:random_index + 1]
            yi = y[random_index]
            
            prediction = np.dot(xi, theta)
            gradient = np.dot(xi.T, (prediction - yi))
            theta -= lr * gradient.flatten()
        
        predictions = np.dot(x, theta)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        cost_history.append(cost)
        
    return theta, cost_history

# Define function for Mini-Batch Gradient Descent
def mini_batch_gradient_descent(x, y, lr, max_iter, batch_size):
    m = len(y)
    theta = np.zeros(x.shape[1])
    cost_history = []
    
    for _ in range(max_iter):
        shuffled_indices = np.random.permutation(m)
        x_shuffled = x[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        for i in range(0, m, batch_size):
            xi_batch = x_shuffled[i:i + batch_size]
            yi_batch = y_shuffled[i:i + batch_size]
            
            predictions = np.dot(xi_batch, theta)
            gradient = np.dot(xi_batch.T, (predictions - yi_batch)) / len(yi_batch)
            theta -= lr * gradient
        
        predictions = np.dot(x, theta)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        cost_history.append(cost)
        
    return theta, cost_history

# Gradient descent parameters
learning_rate = 0.5
max_iterations = 50
batch_size = 16

# Perform gradient descent with each method
theta_bgd, cost_bgd = batch_gradient_descent(x, y, learning_rate, max_iterations)
theta_sgd, cost_sgd = stochastic_gradient_descent(x, y, learning_rate, max_iterations)
theta_mbgd, cost_mbgd = mini_batch_gradient_descent(x, y, learning_rate, max_iterations, batch_size)

# Plot the cost history for all methods
plt.figure(figsize=(10, 6))
plt.plot(cost_bgd, label="Batch Gradient Descent", color='blue')
plt.plot(cost_sgd, label="Stochastic Gradient Descent", color='red')
plt.plot(cost_mbgd, label="Mini-Batch Gradient Descent", color='green')
plt.xlabel("Iterations")
plt.ylabel("Cost Function (J)")
plt.title("Cost Function vs Iterations for Different Gradient Descent Methods")
plt.legend()
plt.show()
