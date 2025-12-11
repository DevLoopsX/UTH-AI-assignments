import math

# ========== DATASET ==========
DATASET = [
    [0.5, 0],  # 0.5 giờ học → Rớt
    [1.0, 0],  # 1.0 giờ học → Rớt
    [1.5, 0],  # 1.5 giờ học → Rớt
    [2.0, 0],  # 2.0 giờ học → Rớt
    [2.5, 1],  # 2.5 giờ học → Đậu
    [3.0, 1],  # 3.0 giờ học → Đậu
    [3.5, 1],  # 3.5 giờ học → Đậu
    [4.0, 1]   # 4.0 giờ học → Đậu
]


# ========== CÁC HÀM CHO LOGISTIC REGRESSION ==========

def get_prediction(m, b, x):
    """
    Tính dự đoán sử dụng hàm sigmoid
    σ(z) = 1 / (1 + e^(-z)), với z = m*x + b

    Args:
        m (float): Hệ số góc
        b (float): Hệ số chặn
        x (float): Giá trị đầu vào

    Returns:
        float: Xác suất dự đoán (0-1)
    """
    z = m * x + b
    return 1 / (1 + math.exp(-z))


def get_cost(y, y_hat):
    """
    Tính cost function (Binary Cross-Entropy Loss)
    J = -1/n * Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]

    Args:
        y (list): Nhãn thực tế
        y_hat (list): Xác suất dự đoán

    Returns:
        float: Giá trị cost trung bình
    """
    n = len(y)
    total_cost = 0.0
    for yi, y_hat_i in zip(y, y_hat):
        total_cost += -(yi * math.log(y_hat_i) + (1 - yi) * math.log(1 - y_hat_i))
    return total_cost / n


def get_gradients(m, b, x, y, y_hat):
    """
    Tính gradient cho m và b
    ∂J/∂m = 1/n * Σ[(ŷ - y) * x]
    ∂J/∂b = 1/n * Σ[(ŷ - y)]

    Args:
        m (float): Hệ số góc hiện tại
        b (float): Hệ số chặn hiện tại
        x (list): Danh sách giá trị đầu vào
        y (list): Nhãn thực tế
        y_hat (list): Xác suất dự đoán

    Returns:
        tuple: (dm, db) - Gradient của m và b
    """
    n = len(y)
    dm = (1 / n) * sum((y_hat_i - yi) * xi for y_hat_i, yi, xi in zip(y_hat, y, x))
    db = (1 / n) * sum(y_hat_i - yi for y_hat_i, yi in zip(y_hat, y))
    return dm, db


def get_accuracy(y, y_hat):
    """
    Tính độ chính xác của mô hình

    Args:
        y (list): Nhãn thực tế
        y_hat (list): Xác suất dự đoán

    Returns:
        float: Độ chính xác (0-1)
    """
    correct_predictions = sum((1 if y_hat_i >= 0.5 else 0) == yi for y_hat_i, yi in zip(y_hat, y))
    return correct_predictions / len(y)


def train_logistic_regression(dataset=DATASET, m_init=1.0, b_init=-1.0, iterations=10, learning_rate=1.0):
    """
    Huấn luyện mô hình Logistic Regression bằng Gradient Descent

    Args:
        dataset (list): Dữ liệu huấn luyện [[x, y], ...]
        m_init (float): Giá trị khởi tạo cho m
        b_init (float): Giá trị khởi tạo cho b
        iterations (int): Số lần lặp
        learning_rate (float): Tốc độ học

    Returns:
        tuple: (m, b, costs) - Tham số học được và lịch sử cost
    """
    m = m_init
    b = b_init

    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]

    costs = []

    for it in range(iterations):
        # Forward pass
        y_hat = [get_prediction(m, b, xi) for xi in x]

        # Calculate cost
        cost = get_cost(y, y_hat)
        costs.append(cost)

        # Backward pass
        dm, db = get_gradients(m, b, x, y, y_hat)

        # Update parameters
        m -= learning_rate * dm
        b -= learning_rate * db

    return m, b, costs
