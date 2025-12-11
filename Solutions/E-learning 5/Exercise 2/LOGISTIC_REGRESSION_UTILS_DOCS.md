# 📚 Tài Liệu Module `logistic_regression_utils.py`

## 🎯 Mục Đích

File `logistic_regression_utils.py` là module tiện ích chứa các hàm và dữ liệu dùng chung cho cả Question A và Question B trong Exercise 2. Module này cung cấp implementation hoàn chỉnh của thuật toán Logistic Regression từ đầu (from scratch), không sử dụng thư viện Machine Learning có sẵn.

---

## 📊 Dataset

```python
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
```

**Giải thích:**

Dataset này biểu diễn mối quan hệ giữa số giờ học (cột 1) và kết quả đậu/rớt (cột 2) của 8 sinh viên. Mỗi dòng trong dataset là một mảng gồm 2 phần tử: phần tử thứ nhất `[row[0]]` là số giờ học (feature/đặc trưng), phần tử thứ hai `[row[1]]` là nhãn kết quả với 0 = rớt và 1 = đậu.

**Đặc điểm của dataset:**

Dataset có tính cân bằng với 4 mẫu rớt (0.5-2.0 giờ) và 4 mẫu đậu (2.5-4.0 giờ). Có ranh giới rõ ràng giữa hai lớp, tạo khoảng cách giữa 2.0 và 2.5 giờ, giúp mô hình dễ dàng học được pattern. Dataset phù hợp cho binary classification, đủ đơn giản để minh họa nhưng vẫn thể hiện được đầy đủ cơ chế hoạt động của Logistic Regression.

---

## 🔧 Các Hàm Trong Module

### 1️⃣ Hàm `get_prediction(m, b, x)`

```python
def get_prediction(m, b, x):
    # Sigmoid function
    y = m * x + b
    return 1 / (1 + math.exp(-y))
```

**Mục đích:**

Hàm này thực hiện forward propagation, tính toán xác suất dự đoán cho một giá trị đầu vào x dựa trên tham số m và b.

**Tham số:**

-   **`m`** (float): Hệ số góc (slope/weight) của mô hình, xác định độ dốc của đường sigmoid
-   **`b`** (float): Hệ số chặn (bias/intercept), xác định vị trí dịch chuyển của đường sigmoid
-   **`x`** (float): Giá trị đầu vào cần dự đoán (số giờ học)

**Giá trị trả về:**

Hàm trả về xác suất thuộc lớp 1 (đậu), là số thực trong khoảng (0, 1).

**Cách hoạt động:**

Bước đầu tiên tính giá trị tuyến tính (linear combination):

$$z = m \times x + b$$

Đây là phương trình đường thẳng cơ bản, trong đó m kiểm soát độ dốc và b kiểm soát điểm cắt trục y.

Bước thứ hai áp dụng hàm sigmoid để chuyển đổi z thành xác suất:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Hàm sigmoid "nén" giá trị z (có thể từ âm vô cùng đến dương vô cùng) vào khoảng (0, 1), phù hợp để biểu diễn xác suất.

**Ví dụ sử dụng:**

```python
m, b = 2.0, -4.0
hours = 2.8
probability = get_prediction(m, b, hours)
# z = 2.0 * 2.8 + (-4.0) = 1.6
# sigmoid(1.6) ≈ 0.832
# Xác suất đậu ≈ 83.2%
```

**Lưu ý kỹ thuật:**

Hàm sử dụng `math.exp()` từ thư viện built-in Python thay vì numpy. Do đó, hàm này chỉ xử lý được một giá trị scalar tại một thời điểm, không thể xử lý array/vector như numpy. Nếu cần dự đoán cho nhiều điểm, cần dùng vòng lặp hoặc list comprehension.

---

### 2️⃣ Hàm `get_cost(y, y_hat)`

```python
def get_cost(y, y_hat):
    # Binary cross-entropy
    k = len(y)
    total_cost = 0.0
    for yi, y_hat_i in zip(y, y_hat):
        total_cost += -(yi * math.log(y_hat_i) + (1 - yi) * math.log(1 - y_hat_i))
    return total_cost / k
```

**Mục đích:**

Hàm này tính toán Binary Cross-Entropy Loss, đo lường mức độ sai lệch giữa dự đoán và thực tế. Giá trị cost càng nhỏ thì mô hình càng tốt.

**Tham số:**

-   **`y`** (list): Danh sách nhãn thực tế, mỗi phần tử là 0 hoặc 1. Ví dụ: `[0, 0, 0, 0, 1, 1, 1, 1]`
-   **`y_hat`** (list): Danh sách xác suất dự đoán, mỗi phần tử trong khoảng (0, 1). Ví dụ: `[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]`

**Giá trị trả về:**

Giá trị cost trung bình (float), luôn là số dương. Cost càng nhỏ càng tốt, với cost = 0 là lý tưởng (dự đoán hoàn hảo).

**Công thức toán học:**

$$J = \frac{1}{k} \sum_{i=1}^{k} -\left[ y_i \log(h_i) + (1-y_i) \log(1-h_i) \right]$$

Trong đó:

-   k là số lượng mẫu
-   yi là nhãn thực tế (0 hoặc 1)
-   hi là xác suất dự đoán

**Cách hoạt động:**

Hàm đếm số lượng mẫu `k = len(y)`, sau đó khởi tạo biến tích lũy `total_cost = 0.0`. Vòng lặp duyệt qua từng cặp (yi, y_hat_i) bằng `zip()`:

```python
for yi, y_hat_i in zip(y, y_hat):
```

Với mỗi cặp, tính loss theo công thức:

$$loss_i = -(y_i \log(h_i) + (1-y_i) \log(1-h_i))$$

**Logic của công thức:**

Khi yi = 1 (thực tế đậu): công thức rút gọn thành:

$$-\log(h_i)$$

Nếu hi gần 1 (dự đoán đúng): -log(1) ≈ 0 → cost thấp

Nếu hi gần 0 (dự đoán sai): -log(0) → +∞ → cost rất cao

Khi yi = 0 (thực tế rớt): công thức rút gọn thành:

$$-\log(1-h_i)$$

Nếu hi gần 0 (dự đoán đúng): -log(1) ≈ 0 → cost thấp

Nếu hi gần 1 (dự đoán sai): -log(0) → +∞ → cost rất cao

Cuối cùng, trả về trung bình: `total_cost / k`

**⚠️ Lưu ý quan trọng:**

Hàm này KHÔNG có epsilon (giá trị nhỏ để tránh log(0)). Trong thực tế, điều này có thể gây lỗi nếu y_hat chứa giá trị 0 hoặc 1 chính xác. Nên cải tiến thành:

```python
epsilon = 1e-15
total_cost += -(yi * math.log(y_hat_i + epsilon) +
               (1 - yi) * math.log(1 - y_hat_i + epsilon))
```

---

### 3️⃣ Hàm `get_gradients(m, b, x, y, y_hat)`

```python
def get_gradients(m, b, x, y, y_hat):
    # Calculate gradients
    k = len(y)
    dm = (1 / k) * sum((y_hat_i - yi) * xi for y_hat_i, yi, xi in zip(y_hat, y, x))
    db = (1 / k) * sum(y_hat_i - yi for y_hat_i, yi in zip(y_hat, y))
    return dm, db
```

**Mục đích:**

Hàm này tính đạo hàm (gradient) của hàm cost theo m và b, chỉ ra hướng và mức độ cần điều chỉnh tham số để giảm cost.

**Tham số:**

-   **`m, b`** (float): Tham số hiện tại (không thực sự sử dụng trong hàm, chỉ để tương thích signature)
-   **`x`** (list): Danh sách giá trị features. Ví dụ: `[0.5, 1.0, 1.5, ..., 4.0]`
-   **`y`** (list): Danh sách nhãn thực tế. Ví dụ: `[0, 0, 0, ..., 1]`
-   **`y_hat`** (list): Danh sách xác suất dự đoán. Ví dụ: `[h1, h2, ..., h8]`

**Giá trị trả về:**

Tuple gồm 2 giá trị: `(dm, db)` - gradient của m và b.

**Công thức toán học:**

Gradient của m (hệ số góc):

$$\frac{\partial J}{\partial m} = \frac{1}{k}\sum_{i=1}^{k}(h_i - y_i) \cdot x_i$$

Gradient của b (hệ số chặn):

$$\frac{\partial J}{\partial b} = \frac{1}{k}\sum_{i=1}^{k}(h_i - y_i)$$

**Cách hoạt động:**

Để tính dm, hàm sử dụng generator expression Pythonic:

```python
dm = (1 / k) * sum((y_hat_i - yi) * xi for y_hat_i, yi, xi in zip(y_hat, y, x))
```

Biểu thức này duyệt đồng thời qua 3 list (y_hat, y, x), tính sai số `(y_hat_i - yi)` nhân với feature `xi`, sau đó tổng hợp và lấy trung bình.

Tương tự cho db nhưng không nhân với xi:

```python
db = (1 / k) * sum(y_hat_i - yi for y_hat_i, yi in zip(y_hat, y))
```

**Ý nghĩa:**

Gradient dương: tham số cần giảm để giảm cost

Gradient âm: tham số cần tăng để giảm cost

Độ lớn gradient: cho biết mức độ cần điều chỉnh

**Ví dụ tính toán:**

```python
x = [0.5, 1.0, 1.5, 2.0]
y = [0, 0, 0, 1]
y_hat = [0.1, 0.2, 0.3, 0.8]

errors = [0.1, 0.2, 0.3, -0.2]
weighted = [0.1*0.5, 0.2*1.0, 0.3*1.5, -0.2*2.0]
         = [0.05, 0.2, 0.45, -0.4]
sum = 0.3
dm = 0.3 / 4 = 0.075
```

---

### 4️⃣ Hàm `get_accuracy(y, y_hat)`

```python
def get_accuracy(y, y_hat):
    correct_predictions = sum((1 if y_hat_i >= 0.5 else 0) == yi for y_hat_i, yi in zip(y_hat, y))
    return correct_predictions / len(y)
```

**Mục đích:**

Hàm này tính độ chính xác (accuracy) của mô hình, là tỷ lệ phần trăm dự đoán đúng.

**Tham số:**

-   **`y`** (list): Nhãn thực tế
-   **`y_hat`** (list): Xác suất dự đoán

**Giá trị trả về:**

Accuracy (float) từ 0.0 (0%) đến 1.0 (100%).

**Cách hoạt động:**

Bước 1: Chuyển xác suất thành nhãn dự đoán với ngưỡng 0.5:

```python
1 if y_hat_i >= 0.5 else 0
```

Bước 2: So sánh với nhãn thực tế:

```python
(... == yi)
```

Kết quả là True (đúng) hoặc False (sai).

Bước 3: Đếm số dự đoán đúng bằng `sum()`. Python tự động chuyển True=1, False=0.

Bước 4: Tính tỷ lệ:

$$Accuracy = \frac{\text{S\u1ed1 d\u1ef1 \u0111o\u00e1n \u0111\u00fang}}{\text{T\u1ed5ng s\u1ed1 m\u1eabu}}$$

**Ví dụ:**

```python
y = [0, 0, 1, 1, 1]
y_hat = [0.2, 0.6, 0.7, 0.8, 0.3]

# Chuyển thành nhãn:
predictions = [0, 1, 1, 1, 0]

# So sánh:
# 0 == 0 ✓
# 1 == 0 ✗
# 1 == 1 ✓
# 1 == 1 ✓
# 0 == 1 ✗

accuracy = 3 / 5 = 0.6 = 60%
```

---

### 5️⃣ Hàm `train_logistic_regression(...)`

```python
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
```

**Mục đích:**

Đây là hàm chính thực hiện thuật toán Gradient Descent, huấn luyện mô hình Logistic Regression từ đầu đến cuối.

**Tham số:**

-   **`dataset`** (list, mặc định = DATASET): Dữ liệu training dạng `[[x1, y1], [x2, y2], ...]`
-   **`m_init`** (float, mặc định = 1.0): Giá trị khởi tạo cho hệ số góc
-   **`b_init`** (float, mặc định = -1.0): Giá trị khởi tạo cho hệ số chặn
-   **`iterations`** (int, mặc định = 10): Số vòng lặp training
-   **`learning_rate`** (float, mặc định = 1.0): Tốc độ học

**Giá trị trả về:**

Tuple gồm 3 phần tử: `(m, b, costs)`

-   **m**: Hệ số góc tối ưu sau training
-   **b**: Hệ số chặn tối ưu sau training
-   **costs**: List chứa giá trị cost tại mỗi iteration

**Các bước thực hiện:**

**Bước 1: Khởi tạo tham số**

```python
m = m_init
b = b_init
```

**Bước 2: Tách dataset thành x và y**

```python
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
```

Kết quả:

```python
x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y = [0, 0, 0, 0, 1, 1, 1, 1]
```

**Bước 3: Khởi tạo list lưu cost history**

```python
costs = []
```

**Bước 4: Vòng lặp training (Gradient Descent)**

```python
for it in range(iterations):
```

Trong mỗi iteration:

_4.1. Forward Propagation:_

```python
y_hat = [get_prediction(m, b, xi) for xi in x]
```

Tính xác suất dự đoán cho tất cả điểm.

_4.2. Tính Cost:_

```python
cost = get_cost(y, y_hat)
costs.append(cost)
```

Đánh giá chất lượng mô hình hiện tại và lưu vào history.

_4.3. Backward Propagation (tính gradient):_

```python
dm, db = get_gradients(m, b, x, y, y_hat)
```

Tính hướng và mức độ cần điều chỉnh.

_4.4. Cập nhật tham số (Gradient Descent step):_

```python
m -= learning_rate * dm
b -= learning_rate * db
```

Công thức:

$$m_{new} = m_{old} - \alpha \frac{\partial J}{\partial m}$$

$$b_{new} = b_{old} - \alpha \frac{\partial J}{\partial b}$$

**Bước 5: Trả về kết quả**

```python
return m, b, costs
```

**Ví dụ sử dụng:**

```python
# Training với default parameters
m, b, costs = train_logistic_regression()

# Training tùy chỉnh
m, b, costs = train_logistic_regression(
    dataset=DATASET,
    m_init=0.0,
    b_init=0.0,
    iterations=50,
    learning_rate=0.5
)

print(f"Optimal m: {m:.4f}")
print(f"Optimal b: {b:.4f}")
print(f"Final cost: {costs[-1]:.4f}")
```

**Quan sát cost convergence:**

```python
import matplotlib.pyplot as plt

plt.plot(range(len(costs)), costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()
```

Cost nên giảm dần qua các iteration. Nếu cost tăng hoặc dao động, có thể learning rate quá lớn.

---

## 📝 So Sánh Với Exercise 1

| Đặc điểm           | Exercise 1    | Exercise 2 (Module này) |
| ------------------ | ------------- | ----------------------- |
| Thư viện           | NumPy         | math (built-in)         |
| Style code         | Vectorized    | Loop-based              |
| Dataset size       | 6 mẫu         | 8 mẫu                   |
| Epsilon trong cost | Có (1e-15)    | Không có ⚠️             |
| Print progress     | Không         | Không                   |
| Return values      | w, b, 3 lists | m, b, 1 list            |
| Tốc độ             | Nhanh hơn     | Chậm hơn                |

**Ưu điểm của module này:**

-   Không phụ thuộc vào thư viện ngoài (chỉ dùng math built-in)
-   Code dễ hiểu, từng bước rõ ràng
-   Phù hợp cho mục đích học tập

**Nhược điểm:**

-   Thiếu epsilon → có thể lỗi với log(0)
-   Chậm hơn NumPy với dataset lớn
-   Chỉ xử lý được scalar, không xử lý được batch

---

## 🔄 Luồng Sử Dụng Điển Hình

```python
# 1. Import module
from logistic_regression_utils import (
    DATASET,
    get_prediction,
    train_logistic_regression
)

# 2. Training
m, b, costs = train_logistic_regression(
    dataset=DATASET,
    iterations=10,
    learning_rate=1.0
)

# 3. Prediction
hours_input = 2.8
probability = get_prediction(m, b, hours_input)

# 4. Classification
if probability >= 0.5:
    result = "ĐẬU"
else:
    result = "RỚT"

print(f"Xác suất: {probability:.4f} → {result}")
```

---