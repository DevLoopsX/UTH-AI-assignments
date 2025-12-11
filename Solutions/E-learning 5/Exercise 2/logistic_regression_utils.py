import math

# Dataset
DATASET = [
    [0.5, 0],
    [1.0, 0],
    [1.5, 0],
    [2.0, 0],
    [2.5, 1],
    [3.0, 1],
    [3.5, 1],
    [4.0, 1]
]

def get_prediction(m, b, x):
    # Sigmoid function
    y = m * x + b
    return 1 / (1 + math.exp(-y))

def get_cost(y, y_hat):
    # Binary cross-entropy
    k = len(y)
    total_cost = 0.0
    for yi, y_hat_i in zip(y, y_hat):
        total_cost += -(yi * math.log(y_hat_i) + (1 - yi) * math.log(1 - y_hat_i))
    return total_cost / k

def get_gradients(m, b, x, y, y_hat):
    # Calculate gradients
    k = len(y)
    dm = (1 / k) * sum((y_hat_i - yi) * xi for y_hat_i, yi, xi in zip(y_hat, y, x))
    db = (1 / k) * sum(y_hat_i - yi for y_hat_i, yi in zip(y_hat, y))
    return dm, db

def get_accuracy(y, y_hat):
    correct_predictions = sum((1 if y_hat_i >= 0.5 else 0) == yi for y_hat_i, yi in zip(y_hat, y))
    return correct_predictions / len(y)

def train_logistic_regression(dataset=DATASET, m_init=1.0, b_init=-1.0, iterations=10, learning_rate=1.0):
    m = m_init
    b = b_init

    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]

    costs = []

    for it in range(iterations):
        y_hat = [get_prediction(m, b, xi) for xi in x]

        cost = get_cost(y, y_hat)
        costs.append(cost)

        dm, db = get_gradients(m, b, x, y, y_hat)

        m -= learning_rate * dm
        b -= learning_rate * db

    return m, b, costs
