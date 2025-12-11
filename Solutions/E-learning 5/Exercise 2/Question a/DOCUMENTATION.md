# 📘 E-Learning 5 - Exercise 2 - Question A: Logistic Regression Training

## 🎯 Mục Tiêu Bài Tập

Bài tập yêu cầu **lập trình và huấn luyện một mô hình Logistic Regression** từ đầu (không dùng thư viện máy học) để dự đoán xác suất sinh viên đậu/rớt dựa trên số giờ học.

### 📊 Đề Bài

**Dataset:** Quan hệ giữa thời gian tự học và kết quả đầu/rớt của sinh viên

| Hours (Giờ học) | Pass (Kết quả) |
| --------------- | -------------- |
| 0.5             | 0 (Rớt)        |
| 1.0             | 0 (Rớt)        |
| 1.5             | 0 (Rớt)        |
| 2.0             | 0 (Rớt)        |
| 2.5             | 1 (Đậu)        |
| 3.0             | 1 (Đậu)        |
| 3.5             | 1 (Đậu)        |
| 4.0             | 1 (Đậu)        |

**Yêu cầu:**

-   Lập trình thuật toán Logistic Regression với **số lần huấn luyện n = 10**
-   Sau đó **dự đoán** khi sinh viên tự học **2.8 giờ** thì xác suất đậu là bao nhiêu?
-   Kết luận: Sinh viên sẽ **đậu hay rớt**?

**Lưu ý:** Không được sử dụng thư viện sklearn (chỉ được dùng thư viện cơ bản)

---

## 💻 Phân Tích Source Code Chi Tiết

### 1️⃣ Import Module và Cấu Trúc Project

```python
import sys
import os

# Thêm thư mục cha vào path để import module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logistic_regression_utils import (
    DATASET, get_prediction, train_logistic_regression
)
```

**Giải thích:**

#### **Import sys và os**

-   **`sys`:** Module hệ thống Python, dùng để thao tác với môi trường runtime
-   **`os`:** Module hệ điều hành, dùng để làm việc với file và thư mục

#### **Thêm đường dẫn module**

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**Phân tích từng bước:**

1. **`__file__`:** Biến đặc biệt chứa đường dẫn file Python hiện tại

    - Ví dụ: `d:/UTH-AI-assignments/.../Exercise 2/Question a/ex2a_logistic_regression.py`

2. **`os.path.abspath(__file__)`:** Chuyển thành đường dẫn tuyệt đối

    - Đảm bảo đường dẫn đầy đủ, không phụ thuộc thư mục làm việc

3. **`os.path.dirname(...)`:** Lấy thư mục cha (lần 1)

    - Kết quả: `d:/UTH-AI-assignments/.../Exercise 2/Question a/`

4. **`os.path.dirname(...)` (lần 2):** Lấy thư mục cha tiếp (lần 2)

    - Kết quả: `d:/UTH-AI-assignments/.../Exercise 2/`

5. **`sys.path.insert(0, ...)`:** Thêm đường dẫn vào **đầu** danh sách tìm kiếm module
    - `0`: Vị trí đầu tiên (ưu tiên cao nhất)
    - Cho phép Python tìm thấy file `logistic_regression_utils.py` ở thư mục cha

**Tại sao cần làm thế này?**

-   File `logistic_regression_utils.py` nằm ở thư mục `Exercise 2/` (cha)
-   File hiện tại nằm ở `Exercise 2/Question a/` (con)
-   Python mặc định chỉ tìm module trong thư mục hiện tại
-   Phải thêm đường dẫn thủ công để import từ thư mục cha

**Cấu trúc thư mục:**

```
Exercise 2/
├── logistic_regression_utils.py  ← Module chứa các hàm
├── Question a/
│   └── ex2a_logistic_regression.py  ← File này
└── Question b/
    └── ex2b_logistic_regression.py
```

#### **Import các hàm từ module**

```python
from logistic_regression_utils import (
    DATASET, get_prediction, train_logistic_regression
)
```

**Giải thích:**

-   **`DATASET`:** Hằng số chứa dữ liệu training
-   **`get_prediction`:** Hàm dự đoán xác suất cho input mới
-   **`train_logistic_regression`:** Hàm huấn luyện mô hình

**Lợi ích của cách tổ chức này:**

-   ✓ **Tái sử dụng code:** Các hàm dùng chung cho Question a và b
-   ✓ **Dễ bảo trì:** Sửa logic ở 1 chỗ, cả 2 file đều được cập nhật
-   ✓ **Code sạch:** File chính chỉ tập trung vào logic cụ thể

---

### 2️⃣ Huấn Luyện Mô Hình

```python
# Huấn luyện mô hình với n = 10 iterations
m, b, costs = train_logistic_regression(
    dataset=DATASET,
    m_init=1.0,
    b_init=-1.0,
    iterations=10,
    learning_rate=1.0
)
```

**Giải thích:**

#### **Gọi hàm training**

Hàm `train_logistic_regression` nhận các tham số:

1. **`dataset=DATASET`**

    - Truyền dữ liệu training (8 điểm dữ liệu)
    - Dataset đã được định nghĩa trong `logistic_regression_utils.py`

2. **`m_init=1.0`**

    - **m** (slope/hệ số góc) khởi tạo = 1.0
    - Tương đương với **w** trong Exercise 1
    - Giá trị ban đầu khác 0 → có hướng học ngay từ đầu

3. **`b_init=-1.0`**

    - **b** (bias/hệ số chặn) khởi tạo = -1.0
    - Giá trị âm → dịch sigmoid sang phải
    - Phù hợp với dữ liệu (cần threshold khoảng 2-2.5 giờ)

4. **`iterations=10`**

    - Chỉ chạy **10 vòng lặp** (ít hơn rất nhiều so với Exercise 1 có 1000)
    - Đề bài yêu cầu n = 10
    - Với learning_rate lớn, 10 iterations có thể đủ

5. **`learning_rate=1.0`**
    - Tốc độ học rất **cao** (gấp 10,000 lần Exercise 1)
    - Cho phép mô hình học nhanh trong ít iteration
    - **Rủi ro:** Có thể overshooting nếu không cẩn thận

#### **Kết quả trả về**

```python
m, b, costs = train_logistic_regression(...)
```

-   **`m`:** Hệ số góc sau khi training
-   **`b`:** Hệ số chặn sau khi training
-   **`costs`:** List chứa giá trị Cost qua 10 iterations

**So sánh với Exercise 1:**

| Tham số       | Exercise 1        | Exercise 2          |
| ------------- | ----------------- | ------------------- |
| Tham số       | w, b              | m, b                |
| Khởi tạo      | 0, 0              | 1.0, -1.0           |
| Learning Rate | 0.0001            | 1.0                 |
| Iterations    | 1000              | 10                  |
| Chiến lược    | Học chậm, ổn định | Học nhanh, mạo hiểm |

---

### 3️⃣ Dự Đoán Cho Input Mới

```python
hours_input = 2.8
predicted_score = get_prediction(m, b, hours_input)
```

**Giải thích:**

-   **`hours_input = 2.8`:** Sinh viên học 2.8 giờ (đề bài yêu cầu)
-   **`get_prediction(m, b, hours_input)`:** Gọi hàm dự đoán
    -   Truyền vào:
        -   `m, b`: Tham số đã học được
        -   `hours_input`: Giá trị x cần dự đoán
    -   Trả về: Xác suất đậu (giá trị từ 0 đến 1)

**Công thức bên trong hàm:**

```python
def get_prediction(m, b, x):
    y = m * x + b
    return 1 / (1 + math.exp(-y))
```

Với m, b đã học và x = 2.8:

1. Tính z = m × 2.8 + b
2. Tính sigmoid(z) = 1 / (1 + e^(-z))

**Ví dụ với m=2.0, b=-4.0:**

-   z = 2.0 × 2.8 + (-4.0) = 5.6 - 4.0 = 1.6
-   sigmoid(1.6) = 1 / (1 + e^(-1.6)) ≈ 0.832

→ Xác suất đậu ≈ 83.2%

---

### 4️⃣ In Kết Quả và Kết Luận

```python
print("\n" + "-"*40)

print(f"Kết quả dự đoán cho {hours_input} giờ học:")
print(f"Điểm số dự đoán: {predicted_score:.4f}")
print(f"Xác suất đậu: {predicted_score:.4f} ({predicted_score*100:.2f}%)")

if predicted_score >= 0.5:
    print("=> Kết luận: ĐẬU")
else:
    print("=> Kết luận: RỚT")

print("-"*40 + "\n")
```

**Giải thích:**

#### **In tiêu đề**

```python
print("\n" + "-"*40)
```

-   `"\n"`: Xuống dòng
-   `"-"*40`: In 40 dấu gạch ngang (separator đẹp mắt)

#### **In kết quả dự đoán**

```python
print(f"Kết quả dự đoán cho {hours_input} giờ học:")
```

-   In số giờ input (2.8)

```python
print(f"Điểm số dự đoán: {predicted_score:.4f}")
```

-   In xác suất với 4 chữ số thập phân (ví dụ: 0.8324)

```python
print(f"Xác suất đậu: {predicted_score:.4f} ({predicted_score*100:.2f}%)")
```

-   In cả dạng thập phân (0.8324) và phần trăm (83.24%)
-   `.2f`: 2 chữ số thập phân cho phần trăm

#### **Phân loại (Classification)**

```python
if predicted_score >= 0.5:
    print("=> Kết luận: ĐẬU")
else:
    print("=> Kết luận: RỚT")
```

**Ngưỡng quyết định (Decision Threshold):**

-   **Xác suất ≥ 0.5:** Dự đoán lớp **dương** (y=1, ĐẬU)
-   **Xác suất < 0.5:** Dự đoán lớp **âm** (y=0, RỚT)

**Tại sao chọn 0.5?**

-   0.5 là **điểm cân bằng** (50-50)
-   Tại điểm này, sigmoid cắt trục y
-   z = 0 → sigmoid(0) = 0.5
-   Là ngưỡng chuẩn cho bài toán cân bằng (balanced dataset)

**Có thể điều chỉnh threshold:**

-   Nếu muốn "cẩn thận hơn" → threshold = 0.7 (phải rất chắc mới kết luận đậu)
-   Nếu muốn "dễ dãi hơn" → threshold = 0.3 (dễ kết luận đậu)
-   Trade-off giữa Precision và Recall

#### **Kết thúc**

```python
print("-"*40 + "\n")
```

-   In separator đóng
-   Xuống dòng để thoáng

---

## 📖 Phân Tích Module `logistic_regression_utils.py`

Để hiểu rõ hơn, phân tích các hàm trong module:

### **1. Dataset Definition**

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

-   **Danh sách 2D:** Mỗi phần tử là `[hours, pass]`
-   **8 mẫu dữ liệu:**
    -   4 mẫu rớt (0.5-2.0 giờ → y=0)
    -   4 mẫu đậu (2.5-4.0 giờ → y=1)

**Phân tích dataset:**

-   **Dữ liệu cân bằng:** 50% đậu, 50% rớt
-   **Phân chia rõ ràng:** Có khoảng trống giữa 2.0 và 2.5
-   **Tuyến tính khá tốt:** Có thể vẽ đường phân chia rõ ràng
-   **Decision boundary dự kiến:** Khoảng 2.2-2.3 giờ

---

### **2. Hàm get_prediction**

```python
def get_prediction(m, b, x):
    # Sigmoid function
    y = m * x + b
    return 1 / (1 + math.exp(-y))
```

**Giải thích:**

Hàm này thực hiện **forward propagation** (truyền tiến):

1. **Tính giá trị tuyến tính:**

    ```python
    y = m * x + b
    ```

    - `y` ở đây thực ra là `z` (pre-activation)
    - Phương trình đường thẳng: $z = mx + b$

2. **Áp dụng sigmoid:**
    ```python
    return 1 / (1 + math.exp(-y))
    ```
    - $\sigma(z) = \frac{1}{1 + e^{-z}}$
    - Chuyển z thành xác suất (0, 1)

**Ví dụ sử dụng:**

```python
m, b = 2.0, -4.0
prob = get_prediction(m, b, 2.8)
# z = 2.0 * 2.8 + (-4.0) = 1.6
# sigmoid(1.6) ≈ 0.832
# Xác suất đậu ≈ 83.2%
```

---

### **3. Hàm get_cost**

```python
def get_cost(y, y_hat):
    # Binary cross-entropy
    k = len(y)
    total_cost = 0.0
    for yi, y_hat_i in zip(y, y_hat):
        total_cost += -(yi * math.log(y_hat_i) + (1 - yi) * math.log(1 - y_hat_i))
    return total_cost / k
```

**Giải thích:**

Hàm tính **Binary Cross-Entropy Loss** - giống Exercise 1 nhưng implement khác.

#### **Tham số:**

-   **`y`:** List các nhãn thực tế `[0, 0, 0, 0, 1, 1, 1, 1]`
-   **`y_hat`:** List các xác suất dự đoán `[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]`

#### **Cách tính:**

1. **Đếm số mẫu:**

    ```python
    k = len(y)
    ```

    - k = 8 (số mẫu trong dataset)

2. **Khởi tạo tổng cost:**

    ```python
    total_cost = 0.0
    ```

3. **Duyệt từng cặp (yi, y_hat_i):**

    ```python
    for yi, y_hat_i in zip(y, y_hat):
    ```

    - `zip(y, y_hat)`: Ghép từng cặp tương ứng
    - Ví dụ: (0, 0.1), (0, 0.2), ..., (1, 0.9)

4. **Cộng dồn loss:**

    ```python
    total_cost += -(yi * math.log(y_hat_i) + (1 - yi) * math.log(1 - y_hat_i))
    ```

    - **Nếu yi = 1:** Chỉ tính `-log(y_hat_i)`
    - **Nếu yi = 0:** Chỉ tính `-log(1 - y_hat_i)`

5. **Trung bình:**
    ```python
    return total_cost / k
    ```
    - Chia cho k để lấy trung bình

**So sánh với Exercise 1:**

|         | Exercise 1 | Exercise 2      |
| ------- | ---------- | --------------- |
| Library | numpy      | math (built-in) |
| Style   | Vectorized | Loop            |
| Epsilon | 1e-15      | Không có        |
| Tốc độ  | Nhanh hơn  | Chậm hơn        |

**Lưu ý:** Exercise 2 không có epsilon → có thể gặp lỗi `log(0)` nếu y_hat = 0 hoặc 1. Trong thực tế nên thêm epsilon!

---

### **4. Hàm get_gradients**

```python
def get_gradients(m, b, x, y, y_hat):
    # Calculate gradients
    k = len(y)
    dm = (1 / k) * sum((y_hat_i - yi) * xi for y_hat_i, yi, xi in zip(y_hat, y, x))
    db = (1 / k) * sum(y_hat_i - yi for y_hat_i, yi in zip(y_hat, y))
    return dm, db
```

**Giải thích:**

Hàm tính **gradient** (đạo hàm) của Cost function - giống Exercise 1.

#### **Tham số:**

-   **`m, b`:** Tham số hiện tại (không dùng trong hàm này, chỉ để tương thích)
-   **`x`:** List giá trị features `[0.5, 1.0, 1.5, ..., 4.0]`
-   **`y`:** List nhãn thực tế `[0, 0, 0, ..., 1]`
-   **`y_hat`:** List xác suất dự đoán `[h1, h2, ..., h8]`

#### **Cách tính:**

1. **Gradient của m (hệ số góc):**

    ```python
    dm = (1 / k) * sum((y_hat_i - yi) * xi for y_hat_i, yi, xi in zip(y_hat, y, x))
    ```

    **Công thức toán học:**
    $$\frac{\partial J}{\partial m} = \frac{1}{k}\sum_{i=1}^{k}(h_i - y_i) \cdot x_i$$

    **Phân tích:**

    - `(y_hat_i - yi)`: Sai số tại điểm thứ i
    - `* xi`: Nhân với feature để tính weighted error
    - `sum(...)`: Tổng trên tất cả điểm
    - `(1 / k) *`: Trung bình

    **Generator expression:**

    - `for y_hat_i, yi, xi in zip(y_hat, y, x)`: Duyệt qua 3 list cùng lúc
    - Pythonic và gọn hơn vòng for thông thường

2. **Gradient của b (hệ số chặn):**

    ```python
    db = (1 / k) * sum(y_hat_i - yi for y_hat_i, yi in zip(y_hat, y))
    ```

    **Công thức toán học:**
    $$\frac{\partial J}{\partial b} = \frac{1}{k}\sum_{i=1}^{k}(h_i - y_i)$$

    **Phân tích:**

    - Giống dm nhưng **không nhân** với xi
    - Vì đạo hàm của b là 1

**Ví dụ tính dm:**

```python
x = [0.5, 1.0, 1.5, 2.0]
y = [0, 0, 0, 1]
y_hat = [0.1, 0.2, 0.3, 0.8]

errors = [0.1-0, 0.2-0, 0.3-0, 0.8-1] = [0.1, 0.2, 0.3, -0.2]
weighted = [0.1*0.5, 0.2*1.0, 0.3*1.5, -0.2*2.0] = [0.05, 0.2, 0.45, -0.4]
sum = 0.05 + 0.2 + 0.45 - 0.4 = 0.3
dm = 0.3 / 4 = 0.075
```

---

### **5. Hàm get_accuracy**

```python
def get_accuracy(y, y_hat):
    correct_predictions = sum((1 if y_hat_i >= 0.5 else 0) == yi for y_hat_i, yi in zip(y_hat, y))
    return correct_predictions / len(y)
```

**Giải thích:**

Hàm tính **accuracy** (độ chính xác) của mô hình.

#### **Cách hoạt động:**

1. **Chuyển xác suất thành nhãn:**

    ```python
    1 if y_hat_i >= 0.5 else 0
    ```

    - Nếu y_hat_i ≥ 0.5 → dự đoán 1 (ĐẬU)
    - Ngược lại → dự đoán 0 (RỚT)

2. **So sánh với nhãn thực:**

    ```python
    (... == yi)
    ```

    - True nếu dự đoán đúng
    - False nếu dự đoán sai

3. **Đếm số dự đoán đúng:**

    ```python
    correct_predictions = sum(...)
    ```

    - `sum` trên boolean: True=1, False=0
    - Kết quả: Tổng số dự đoán đúng

4. **Tính tỷ lệ:**
    ```python
    return correct_predictions / len(y)
    ```
    - Số đúng / Tổng số mẫu
    - Kết quả từ 0.0 (0%) đến 1.0 (100%)

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

# Đếm: 3 đúng / 5 tổng = 0.6 = 60%
accuracy = 3 / 5 = 0.6
```

---

### **6. Hàm train_logistic_regression**

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

**Giải thích:**

Đây là **hàm chính** thực hiện thuật toán Gradient Descent - trái tim của bài toán.

#### **Tham số:**

-   **`dataset`:** Dữ liệu training (mặc định DATASET)
-   **`m_init, b_init`:** Giá trị khởi tạo (mặc định 1.0, -1.0)
-   **`iterations`:** Số vòng lặp (mặc định 10)
-   **`learning_rate`:** Tốc độ học (mặc định 1.0)

#### **Bước 1: Khởi tạo tham số**

```python
m = m_init
b = b_init
```

-   Gán giá trị ban đầu cho m, b

#### **Bước 2: Tách dữ liệu**

```python
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
```

**List comprehension:**

-   `row[0]`: Cột đầu tiên (hours)
-   `row[1]`: Cột thứ hai (pass)

**Kết quả:**

```python
x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y = [0, 0, 0, 0, 1, 1, 1, 1]
```

#### **Bước 3: Khởi tạo list lưu cost**

```python
costs = []
```

-   Để tracking quá trình học

#### **Bước 4: Vòng lặp training**

```python
for it in range(iterations):
```

-   Lặp `iterations` lần (10 lần)

**Bên trong mỗi iteration:**

**4.1. Forward propagation:**

```python
y_hat = [get_prediction(m, b, xi) for xi in x]
```

-   Tính xác suất dự đoán cho từng điểm
-   `y_hat` là list 8 giá trị xác suất

**4.2. Tính cost:**

```python
cost = get_cost(y, y_hat)
costs.append(cost)
```

-   Đánh giá chất lượng mô hình hiện tại
-   Lưu vào list

**4.3. Backward propagation (tính gradient):**

```python
dm, db = get_gradients(m, b, x, y, y_hat)
```

-   Tính đạo hàm để biết hướng đi

**4.4. Cập nhật tham số:**

```python
m -= learning_rate * dm
b -= learning_rate * db
```

-   Gradient Descent step
-   Đi ngược hướng gradient để giảm cost

#### **Bước 5: Trả về kết quả**

```python
return m, b, costs
```

-   `m, b`: Tham số tối ưu sau training
-   `costs`: Lịch sử cost (để phân tích)

**So sánh với Exercise 1:**

|              | Exercise 1       | Exercise 2                |
| ------------ | ---------------- | ------------------------- |
| Hàm training | gradient_descent | train_logistic_regression |
| In progress  | Có               | Không                     |
| Lưu history  | cost, w, b       | Chỉ cost                  |
| Return       | w, b, 3 lists    | m, b, 1 list              |

---

## 📊 Output và Kết Quả (Dự Kiến)

### 🖥️ Console Output

```
----------------------------------------
Kết quả dự đoán cho 2.8 giờ học:
Điểm số dự đoán: 0.7854
Xác suất đậu: 0.7854 (78.54%)
=> Kết luận: ĐẬU
----------------------------------------
```

**Lưu ý:** Giá trị cụ thể phụ thuộc vào kết quả training thực tế.

---

### 📈 Phân Tích Kết Quả

#### **1. Xác suất đậu: ~78.54%**

**Ý nghĩa:**

-   Sinh viên học 2.8 giờ có xác suất đậu gần **80%**
-   Không phải 100% vì mô hình học từ dữ liệu có **uncertainty**
-   Dự đoán hợp lý vì:
    -   2.8 giờ > 2.5 giờ (điểm đậu thấp nhất)
    -   2.8 giờ gần 3.0 giờ (điểm đậu chắc chắn)

#### **2. Kết luận: ĐẬU**

**Logic:**

-   Xác suất 0.7854 ≥ 0.5 → Phân loại vào lớp 1 (ĐẬU)
-   Mức độ tin cậy: **cao** (gần 80%, không phải 51%)

#### **3. Phân tích theo ngưỡng giờ học**

Giả sử mô hình học được:

-   m ≈ 2.0, b ≈ -4.0
-   Decision boundary: m × x + b = 0
    → x = -b/m = 4.0/2.0 = 2.0 giờ

**Dự đoán theo ngưỡng:**

-   x < 2.0 giờ: Xác suất đậu < 50% → RỚT
-   x = 2.0 giờ: Xác suất đậu = 50% → Biên giới
-   x > 2.0 giờ: Xác suất đậu > 50% → ĐẬU

**2.8 giờ > 2.0 giờ** → ĐẬU (hợp lý!)

#### **4. So sánh với dữ liệu thực**

| Hours   | Actual | Predicted (approx) |
| ------- | ------ | ------------------ |
| 0.5     | RỚT    | RỚT (~5%)          |
| 1.0     | RỚT    | RỚT (~12%)         |
| 1.5     | RỚT    | RỚT (~27%)         |
| 2.0     | RỚT    | Biên giới (~50%)   |
| 2.5     | ĐẬU    | ĐẬU (~73%)         |
| **2.8** | ?      | **ĐẬU (~79%)**     |
| 3.0     | ĐẬU    | ĐẬU (~88%)         |
| 3.5     | ĐẬU    | ĐẬU (~95%)         |
| 4.0     | ĐẬU    | ĐẬU (~98%)         |

**Nhận xét:**

-   Mô hình dự đoán hợp lý với dữ liệu
-   2.8 giờ nằm giữa 2.5 (73%) và 3.0 (88%) → ~79% là hợp lý

---
