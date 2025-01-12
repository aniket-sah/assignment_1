from utils.csv_manager import Reader
import matplotlib.pyplot as plt

csv_reader = Reader("dataset/linearX.csv", "dataset/linearY.csv")
[x, y] = csv_reader.load_csv()

x_min, x_max = min(x), max(x)
x = [(val - x_min) / (x_max - x_min) for val in x]

LEARNING_RATES = [0.005, 0.5, 5]
FEATURE_COUNT = min(len(x), len(y))
MAX_ITERATIONS = 10000

for lr in LEARNING_RATES:
    iterations = 0
    theta0 = theta1 = 0
    last_cost = float("inf")
    costs = []

    while True:
        iterations += 1
        predictions = [theta0 + theta1 * x[i] for i in range(FEATURE_COUNT)]
        grad0 = (1 / FEATURE_COUNT) * sum([(predictions[i] - y[i]) for i in range(FEATURE_COUNT)])
        grad1 = (1 / FEATURE_COUNT) * sum([(predictions[i] - y[i]) * x[i] for i in range(FEATURE_COUNT)])
        theta0 -= lr * grad0
        theta1 -= lr * grad1
        current_cost = (1 / (2 * FEATURE_COUNT)) * sum([(predictions[i] - y[i]) ** 2 for i in range(FEATURE_COUNT)])
        if iterations <= 50:
            costs.append(current_cost)
        if abs(last_cost - current_cost) < 1e-5 or iterations > MAX_ITERATIONS:
            break
        last_cost = current_cost

    theta1_original = theta1 * (max(y) - min(y)) / (x_max - x_min)
    theta0_original = min(y) + theta0 * (max(y) - min(y)) - theta1_original * x_min

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(costs) + 1), costs, marker='o', linestyle='-', label=f"LR={lr}")
    plt.title(f'Cost Function vs. Iterations (LR={lr})')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend()
    plt.show()

    if lr == 0.5:
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Data Points')
        plt.plot(x, [theta0 + theta1 * xi for xi in x], color='red', label='Regression Line')
        plt.title('Dataset and Regression Line')
        plt.xlabel('Normalized X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    print(f"Learning Rate: {lr}")
    print(f"Final theta0: {theta0}, theta1: {theta1}")
    print(f"Denormalized theta0: {theta0_original}, theta1: {theta1_original}")