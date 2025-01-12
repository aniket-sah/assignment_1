from utils.csv_manager import Reader
import matplotlib.pyplot as plt

# Initialize variables
LEARNING_RATE = 0.5
FEATURE_COUNT = min(len(x), len(y))
MAX_ITERATIONS = 10000

iterations = 0
theta0 = theta1 = 0
last_cost = float("inf")
costs = []  # To store costs for plotting

# Gradient Descent Loop
while True:
    iterations += 1

    # Predictions
    predictions = [theta0 + theta1 * x[i] for i in range(FEATURE_COUNT)]

    # Gradients
    grad0 = (1 / FEATURE_COUNT) * sum([(predictions[i] - y[i]) for i in range(FEATURE_COUNT)])
    grad1 = (1 / FEATURE_COUNT) * sum([(predictions[i] - y[i]) * x[i] for i in range(FEATURE_COUNT)])

    # Update parameters
    theta0 -= LEARNING_RATE * grad0
    theta1 -= LEARNING_RATE * grad1

    # Cost calculation
    current_cost = (1 / (2 * FEATURE_COUNT)) * sum([(predictions[i] - y[i]) ** 2 for i in range(FEATURE_COUNT)])

    # Store cost for plotting
    if iterations <= 50:  # Track only the first 50 iterations
        costs.append(current_cost)

    # Check for convergence
    if abs(last_cost - current_cost) < 1e-5 or iterations > MAX_ITERATIONS:
        break
    last_cost = current_cost

# Plot cost vs iterations
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(costs) + 1), costs, marker='o', linestyle='-', color='b')
plt.title('Cost Function vs. Iterations (First 50 Iterations)')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)
plt.show()

# Denormalize theta values
theta1_original = theta1 * (y_max - y_min) / (x_max - x_min)
theta0_original = y_min + theta0 * (y_max - y_min) - theta1_original * x_min

print("Normalized theta0 and theta1:", theta0, theta1)
print("Denormalized theta0 and theta1:", theta0_original, theta1_original)