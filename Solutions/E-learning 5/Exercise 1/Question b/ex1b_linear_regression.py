import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu
X = np.array([0.5, 1, 1.5, 3, 2, 1])
y = np.array([0, 0, 0, 1, 1, 1])

# Tham số ban đầu
w = 0
b = 0
alpha = 0.0001


# --- Hàm sigmoid ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --- Hàm cost (Binary Cross Entropy) ---
def compute_cost(X, y, w, b):
    m = len(X)
    z = w * X + b
    h = sigmoid(z)
    eps = 1e-15
    cost = -(1/m) * np.sum(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
    return cost


# --- Hàm tính gradient ---
def compute_gradient(X, y, w, b):
    m = len(X)
    z = w * X + b
    h = sigmoid(z)
    error = h - y
    dw = (1/m) * np.sum(error * X)
    db = (1/m) * np.sum(error)
    return dw, db


# --- Gradient Descent ---
def gradient_descent(X, y, w, b, alpha, num_iterations):
    cost_history, w_history, b_history = [], [], []

    for i in range(num_iterations):
        dw, db = compute_gradient(X, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        cost = compute_cost(X, y, w, b)

        cost_history.append(cost)
        w_history.append(w)
        b_history.append(b)

        # In 1 vài vòng lặp quan trọng
        if i == 0 or (i + 1) % 200 == 0 or i == num_iterations - 1:
            print(f"Iteration {i+1:4d} :  w = {w:.6f},  b = {b:.6f},  Cost = {cost:.8f}")

    return w, b, cost_history, w_history, b_history


print("=" * 60)
print("CÂU B – Cập nhật w, b bằng thuật toán Gradient Descent")
print("=" * 60)

initial_cost = compute_cost(X, y, w, b)
print(f"Cost ban đầu (w=0, b=0):  {initial_cost:.8f}\n")

num_iterations = 1000
w_final, b_final, cost_history, w_history, b_history = gradient_descent(
    X, y, w, b, alpha, num_iterations
)

print("\nKẾT QUẢ SAU TRAINING:")
print(f"w_update = {w_final:.8f}")
print(f"b_update = {b_final:.8f}")
print(f"Cost cuối = {cost_history[-1]:.8f}")
print(f"Cost giảm được: {initial_cost - cost_history[-1]:.8f}")

plt.figure(figsize=(8,5))
# --- 1. Đồ thị Cost ---
plt.subplot()
plt.plot(cost_history, 'b', linewidth=2)
plt.title(f"Sự hội tụ của hàm Cost J(w,b) = {cost_history[-1]:.8f}", fontsize=14, fontweight='bold')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('results/ex1b_gradient_descent_convergence.png',
            dpi=300, bbox_inches='tight')
plt.show()
