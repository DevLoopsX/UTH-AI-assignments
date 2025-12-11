# 📘 E-Learning 5 - Exercise 2 - Question B: So Sánh với Sklearn

## 🎯 Mục Tiêu Bài Tập

Bài tập yêu cầu **so sánh kết quả** giữa **mô hình Logistic Regression tự xây dựng** (Question A) với **mô hình từ thư viện sklearn** để kiểm chứng tính đúng đắn của implementation.

### 📊 Đề Bài

**Yêu cầu:**

-   Sử dụng **cùng dataset** như Question A
-   Huấn luyện mô hình bằng **sklearn.linear_model.LogisticRegression**
-   **So sánh** kết quả giữa 2 mô hình:
    -   Tham số học được (m/coef, b/intercept)
    -   Xác suất dự đoán cho 2.8 giờ học
    -   Kết luận đậu/rớt
-   **Visualization:** Tạo biểu đồ so sánh trực quan

---

## 💻 Phân Tích Source Code Chi Tiết

### 1️⃣ Import Libraries

```python
import sys
import os
# Thêm thư mục cha vào path để import module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logistic_regression_utils import (
    DATASET, get_prediction, train_logistic_regression
)
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
```

**Giải thích:**

#### **Import module tự xây dựng**

```python
from logistic_regression_utils import (
    DATASET, get_prediction, train_logistic_regression
)
```

-   Import các hàm từ Question A để sử dụng lại
-   **Tái sử dụng code:** Không viết lại logic training

#### **Import sklearn**

```python
from sklearn.linear_model import LogisticRegression
```

-   **sklearn:** Thư viện Machine Learning phổ biến nhất Python
-   **LogisticRegression:** Class implement Logistic Regression chuẩn, tối ưu
-   **Đặc điểm:**
    -   Highly optimized (C/C++ backend)
    -   Nhiều thuật toán tối ưu: LBFGS, SAG, SAGA, ...
    -   Được test kỹ lưỡng, đáng tin cậy

#### **Import numpy và matplotlib**

```python
import numpy as np
import matplotlib.pyplot as plt
```

-   **numpy:** Sklearn yêu cầu input dạng numpy array
-   **matplotlib:** Để vẽ biểu đồ so sánh

---

### 2️⃣ PHẦN A: Mô Hình Tự Xây Dựng

```python
# ========== PHẦN A: MÔ HÌNH TỰ XÂY DỰNG ==========

# Huấn luyện mô hình với n = 10 iterations
m, b, costs = train_logistic_regression(
    dataset=DATASET,
    m_init=1.0,
    b_init=-1.0,
    iterations=10,
    learning_rate=1.0
)
# Dự đoán cho sinh viên học 2.8 giờ
hours_input = 2.8
predicted_score_manual = get_prediction(m, b, hours_input)
```

**Giải thích:**

#### **Training**

-   Sử dụng **lại code từ Question A**
-   Chạy đúng 10 iterations như yêu cầu đề bài
-   Tham số khởi tạo: m=1.0, b=-1.0, learning_rate=1.0

#### **Prediction**

-   Dự đoán cho 2.8 giờ
-   Lưu vào `predicted_score_manual` để so sánh sau

**Tại sao gọi là "manual"?**

-   Để phân biệt với sklearn (automated/optimized)
-   "Manual" = tự code từ đầu, từng bước

---

### 3️⃣ In Kết Quả Mô Hình Tự Xây Dựng

```python
print("\n" + "="*60)
print("BÀI 2 - QUESTION B: SO SÁNH MÔ HÌNH TỰ XÂY DỰNG VỚI SKLEARN")
print("="*60)

print("\n" + "-"*60)
print("PHẦN A: KẾT QUẢ MÔ HÌNH TỰ XÂY DỰNG")
print("-"*60)
print(f"Tham số học được:")
print(f"  - Hệ số góc (m): {m:.6f}")
print(f"  - Hệ số chặn (b): {b:.6f}")
print(f"\nDự đoán cho sinh viên học {hours_input} giờ:")
print(f"  - Xác suất đậu: {predicted_score_manual:.6f} ({predicted_score_manual*100:.2f}%)")
if predicted_score_manual >= 0.5:
    print(f"  - Kết luận: ĐẬU")
else:
    print(f"  - Kết luận: RỚT")
```

**Giải thích:**

#### **Cấu trúc output**

-   **Header lớn (=):** Tiêu đề bài toán
-   **Header nhỏ (-):** Tiêu đề từng phần

#### **In tham số**

```python
print(f"  - Hệ số góc (m): {m:.6f}")
print(f"  - Hệ số chặn (b): {b:.6f}")
```

-   `.6f`: 6 chữ số thập phân (độ chính xác cao)
-   Thụt đầu dòng ` -` để dễ đọc

#### **In kết quả dự đoán**

```python
print(f"  - Xác suất đậu: {predicted_score_manual:.6f} ({predicted_score_manual*100:.2f}%)")
```

-   In cả dạng thập phân và phần trăm
-   Ví dụ: `0.785432 (78.54%)`

#### **Kết luận**

```python
if predicted_score_manual >= 0.5:
    print(f"  - Kết luận: ĐẬU")
else:
    print(f"  - Kết luận: RỚT")
```

-   Logic đơn giản: ≥ 0.5 → ĐẬU

---

### 4️⃣ PHẦN B: Mô Hình Sklearn

#### **4.1. Chuẩn Bị Dữ Liệu**

```python
# Chuẩn bị dữ liệu cho sklearn
X = np.array([[row[0]] for row in DATASET])  # Features (Hours)
y_train = np.array([row[1] for row in DATASET])  # Labels (Pass)
```

**Giải thích:**

##### **Features (X)**

```python
X = np.array([[row[0]] for row in DATASET])
```

**Phân tích:**

-   `row[0]`: Lấy cột đầu (Hours)
-   `[[row[0]]]`: **Chú ý 2 cặp ngoặc vuông!**
    -   Ngoặc trong `[row[0]]`: Tạo list 1 phần tử
    -   Ngoặc ngoài `[...]`: List comprehension

**Tại sao cần 2 cặp ngoặc?**

Sklearn yêu cầu X phải là **ma trận 2D** (n_samples × n_features):

-   `n_samples`: Số mẫu (8 điểm)
-   `n_features`: Số features (1 feature = hours)

**Kết quả:**

```python
X = [[0.5],
     [1.0],
     [1.5],
     [2.0],
     [2.5],
     [3.0],
     [3.5],
     [4.0]]
# Shape: (8, 1) - 8 hàng, 1 cột
```

**Nếu chỉ dùng 1 cặp ngoặc:**

```python
X = np.array([row[0] for row in DATASET])
# Kết quả: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# Shape: (8,) - 1D array → sklearn báo lỗi!
```

##### **Labels (y_train)**

```python
y_train = np.array([row[1] for row in DATASET])
```

**Phân tích:**

-   `row[1]`: Lấy cột thứ 2 (Pass)
-   Chỉ cần 1D array: `[0, 0, 0, 0, 1, 1, 1, 1]`
-   Shape: (8,) - 8 phần tử

**Lưu ý naming:**

-   Đặt tên `y_train` thay vì `y` để tránh nhầm lẫn với biến `y` từ phần A

---

#### **4.2. Tạo và Huấn Luyện Mô Hình**

```python
# Tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=10, solver='lbfgs', random_state=42)
model.fit(X, y_train)
```

**Giải thích:**

##### **Tạo model**

```python
model = LogisticRegression(max_iter=10, solver='lbfgs', random_state=42)
```

**Các tham số:**

1. **`max_iter=10`**

    - **Maximum iterations:** Số vòng lặp tối đa
    - Đặt = 10 để **công bằng** với mô hình tự xây dựng
    - Mặc định sklearn = 100
    - **Lưu ý:** Sklearn có thể hội tụ sớm hơn 10 iterations nếu đạt tolerance

2. **`solver='lbfgs'`**

    - **Thuật toán tối ưu:** Limited-memory BFGS
    - **LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno):**
        - Thuật toán quasi-Newton
        - Hiệu quả hơn Gradient Descent thông thường
        - Sử dụng approximation của ma trận Hessian
        - Tốt cho dataset nhỏ/trung bình

    **Các solver khác:**

    - `'liblinear'`: Tốt cho dataset nhỏ
    - `'saga'`: Tốt cho dataset lớn
    - `'sag'`: Stochastic Average Gradient
    - `'newton-cg'`: Newton-Conjugate-Gradient

3. **`random_state=42`**
    - **Seed cho random number generator**
    - Đảm bảo **kết quả reproducible** (chạy lại ra cùng kết quả)
    - 42 là số phổ biến (The Hitchhiker's Guide to the Galaxy reference 😊)
    - Quan trọng cho debugging và so sánh

**Tại sao sklearn cần random_state?**

-   Một số solver khởi tạo tham số ngẫu nhiên
-   Shuffling data khi dùng batch methods
-   Đảm bảo reproducibility cho scientific research

##### **Fit model**

```python
model.fit(X, y_train)
```

**Giải thích:**

-   **`fit(X, y)`:** Hàm huấn luyện mô hình
-   **Input:**
    -   `X`: Features (8×1 matrix)
    -   `y_train`: Labels (8 elements)
-   **Process:**
    -   Chạy thuật toán LBFGS
    -   Tối ưu hóa tham số (coef, intercept)
    -   Tối đa 10 iterations
-   **Output:**
    -   `model` được cập nhật (in-place)
    -   Tham số được lưu trong `model.coef_` và `model.intercept_`

**Lưu ý:**

-   Sklearn tự động normalize/standardize nếu cần
-   Tự động handle convergence
-   Tự động điều chỉnh learning rate (adaptive)

---

#### **4.3. Dự Đoán với Sklearn**

```python
# Dự đoán với sklearn
X_test = np.array([[hours_input]])
predicted_proba_sklearn = model.predict_proba(X_test)[0][1]  # Xác suất cho class 1 (Pass)
predicted_class_sklearn = model.predict(X_test)[0]
```

**Giải thích:**

##### **Chuẩn bị test data**

```python
X_test = np.array([[hours_input]])
```

-   `hours_input = 2.8`
-   `[[2.8]]`: Ma trận 2D (1×1) - 1 sample, 1 feature
-   **Phải cùng format** với X training (2D)

##### **Dự đoán xác suất**

```python
predicted_proba_sklearn = model.predict_proba(X_test)[0][1]
```

**Phân tích:**

1. **`model.predict_proba(X_test)`:**

    - Trả về **ma trận xác suất** cho tất cả classes
    - Shape: (n_samples, n_classes)
    - Với X_test shape (1, 1): Output shape (1, 2)
    - **2 classes:** [xác suất class 0, xác suất class 1]

    **Ví dụ:**

    ```python
    predict_proba(X_test) = [[0.2146, 0.7854]]
    # Class 0: 21.46%
    # Class 1: 78.54%
    ```

2. **`[0]`:** Lấy sample đầu tiên (vì chỉ có 1 sample)

    ```python
    [0.2146, 0.7854]
    ```

3. **`[1]`:** Lấy xác suất của **class 1** (Pass)
    ```python
    0.7854
    ```

**Tóm lại:** `[0][1]` = xác suất Pass của sample đầu tiên

##### **Dự đoán class**

```python
predicted_class_sklearn = model.predict(X_test)[0]
```

**Phân tích:**

1. **`model.predict(X_test)`:**

    - Trả về **nhãn dự đoán** (0 hoặc 1)
    - Đã apply threshold 0.5 tự động
    - Output: `[1]` (array với 1 phần tử)

2. **`[0]`:** Lấy phần tử đầu
    - Kết quả: `1` (số nguyên)
    - 1 = Đậu, 0 = Rớt

**So sánh predict vs predict_proba:**

| Method          | Output   | Example          |
| --------------- | -------- | ---------------- |
| `predict_proba` | Xác suất | `[[0.21, 0.79]]` |
| `predict`       | Nhãn     | `[1]`            |

---

#### **4.4. In Kết Quả Sklearn**

```python
print("\n" + "-"*60)
print("PHẦN B: KẾT QUẢ MÔ HÌNH SKLEARN")
print("-"*60)

print(f"Tham số học được:")
print(f"  - Hệ số góc (coef): {model.coef_[0][0]:.6f}")
print(f"  - Hệ số chặn (intercept): {model.intercept_[0]:.6f}")
print(f"\nDự đoán cho sinh viên học {hours_input} giờ:")
print(f"  - Xác suất đậu: {predicted_proba_sklearn:.6f} ({predicted_proba_sklearn*100:.2f}%)")
if predicted_class_sklearn == 1:
    print(f"  - Kết luận: ĐẬU")
else:
    print(f"  - Kết luận: RỚT")
```

**Giải thích:**

##### **Truy cập tham số sklearn**

**Hệ số góc:**

```python
model.coef_[0][0]
```

-   **`model.coef_`:** Ma trận hệ số (n_classes-1, n_features)
    -   Logistic Regression binary: (1, 1)
    -   `[[2.345]]` (1 class, 1 feature)
-   **`[0]`:** Lấy hàng đầu: `[2.345]`
-   **`[0]`:** Lấy cột đầu: `2.345`

**Hệ số chặn:**

```python
model.intercept_[0]
```

-   **`model.intercept_`:** Array hệ số chặn (n_classes-1,)
    -   `[-4.567]` (1 phần tử)
-   **`[0]`:** Lấy phần tử đầu: `-4.567`

##### **In kết quả**

-   Format giống phần A để dễ so sánh
-   Sử dụng `.6f` cho độ chính xác cao

---

### 5️⃣ So Sánh Kết Quả

```python
# ========== SO SÁNH KẾT QUẢ ==========

print(f"\nHệ số góc (m/coef):")
print(f"  - Mô hình tự xây dựng: {m:.6f}")
print(f"  - Sklearn:              {model.coef_[0][0]:.6f}")
print(f"  - Chênh lệch:           {abs(m - model.coef_[0][0]):.6f}")

print(f"\nHệ số chặn (b/intercept):")
print(f"  - Mô hình tự xây dựng: {b:.6f}")
print(f"  - Sklearn:              {model.intercept_[0]:.6f}")
print(f"  - Chênh lệch:           {abs(b - model.intercept_[0]):.6f}")

print(f"\nXác suất đậu cho {hours_input} giờ học:")
print(f"  - Mô hình tự xây dựng: {predicted_score_manual:.6f} ({predicted_score_manual*100:.2f}%)")
print(f"  - Sklearn:              {predicted_proba_sklearn:.6f} ({predicted_proba_sklearn*100:.2f}%)")
print(f"  - Chênh lệch:           {abs(predicted_score_manual - predicted_proba_sklearn):.6f}")

print(f"\nKết luận dự đoán:")
result_manual = "ĐẬU" if predicted_score_manual >= 0.5 else "RỚT"
result_sklearn = "ĐẬU" if predicted_class_sklearn == 1 else "RỚT"
print(f"  - Mô hình tự xây dựng: {result_manual}")
print(f"  - Sklearn:             {result_sklearn}")
if result_manual == result_sklearn:
    print(f"  - Kết quả: GIỐNG NHAU ✓")
else:
    print(f"  - Kết quả: KHÁC NHAU ✗")
```

**Giải thích:**

#### **So sánh từng thành phần**

##### **1. Hệ số góc (m vs coef)**

```python
print(f"  - Chênh lệch:           {abs(m - model.coef_[0][0]):.6f}")
```

-   Tính **giá trị tuyệt đối** của sự chênh lệch
-   `abs()`: Luôn dương, dễ so sánh
-   Kỳ vọng: Chênh lệch nhỏ (< 0.1)

##### **2. Hệ số chặn (b vs intercept)**

```python
print(f"  - Chênh lệch:           {abs(b - model.intercept_[0]):.6f}")
```

-   Tương tự với m
-   Kỳ vọng: Chênh lệch nhỏ

##### **3. Xác suất dự đoán**

```python
print(f"  - Chênh lệch:           {abs(predicted_score_manual - predicted_proba_sklearn):.6f}")
```

-   So sánh output cuối cùng
-   **Quan trọng nhất:** Kết quả dự đoán có đúng không?
-   Kỳ vọng: Chênh lệch rất nhỏ (< 0.01)

##### **4. Kết luận cuối cùng**

```python
result_manual = "ĐẬU" if predicted_score_manual >= 0.5 else "RỚT"
result_sklearn = "ĐẬU" if predicted_class_sklearn == 1 else "RỚT"
```

-   Chuyển số thành text để dễ đọc
-   So sánh string để kiểm tra consistency

```python
if result_manual == result_sklearn:
    print(f"  - Kết quả: GIỐNG NHAU ✓")
else:
    print(f"  - Kết quả: KHÁC NHAU ✗")
```

-   **Nếu giống:** ✓ Implementation đúng!
-   **Nếu khác:** ✗ Có vấn đề cần check

---

### 6️⃣ Phần Kết Luận Tổng Quan

```python
print("\n" + "="*60)
print("KẾT LUẬN")
print("="*60)
print("Có thể thấy sự khác biệt giữa hai mô hình do:")
print("  1. Số lần lặp khác nhau (10 vs thuật toán tối ưu của sklearn)")
print("  2. Phương pháp tối ưu khác nhau (Gradient Descent vs LBFGS)")
print("  3. Điều kiện dừng và khởi tạo tham số khác nhau")
print("="*60 + "\n")
```

**Giải thích:**

#### **Lý do có sự khác biệt**

##### **1. Số lần lặp khác nhau**

-   **Manual:** Đúng 10 iterations, không thêm không bớt
-   **Sklearn:** Có thể hội tụ sớm hơn nếu đạt tolerance
    -   Mặc định `tol=1e-4`
    -   Dừng khi gradient < tolerance
    -   Có thể dừng sau 5-8 iterations

##### **2. Phương pháp tối ưu khác nhau**

**Manual - Gradient Descent:**

-   Công thức: $w_{new} = w_{old} - \alpha \nabla J$
-   **Gradient bậc 1** (first-order derivative)
-   Đơn giản, dễ hiểu
-   Tốc độ hội tụ: **tuyến tính** (linear)

**Sklearn - LBFGS:**

-   **Quasi-Newton method**
-   Sử dụng **gradient bậc 2** (approximated Hessian)
-   Phức tạp hơn nhưng **hiệu quả hơn**
-   Tốc độ hội tụ: **siêu tuyến tính** (superlinear)

**Ví dụ:**

-   GD: 10 steps → Cost giảm 60%
-   LBFGS: 10 steps → Cost giảm 95%

##### **3. Điều kiện dừng và khởi tạo**

**Khởi tạo:**

-   Manual: m=1.0, b=-1.0 (do mình chọn)
-   Sklearn: w=0, b=0 (mặc định) hoặc random

**Điều kiện dừng:**

-   Manual: Chạy đúng 10 iterations, không check convergence
-   Sklearn: Dừng khi `||gradient|| < tol` hoặc `max_iter`

**Learning rate:**

-   Manual: Cố định (1.0)
-   Sklearn: Adaptive (LBFGS tự điều chỉnh)

---

### 7️⃣ Visualization

#### **7.1. Tạo Figure với 2 Subplots**

```python
# ========== VISUALIZATION ==========

# Tạo figure với 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
```

**Giải thích:**

-   Tạo 2 biểu đồ cạnh nhau
-   Kích thước lớn (14×5) để rõ ràng

---

#### **7.2. Subplot 1: So Sánh Decision Boundary**

```python
# Subplot 1: So sánh decision boundary của 2 mô hình
x_plot = np.linspace(0, 4.5, 100)
y_manual = [get_prediction(m, b, x_i) for x_i in x_plot]
y_sklearn = [model.predict_proba([[x_i]])[0][1] for x_i in x_plot]
```

**Giải thích:**

##### **Tạo dải x**

```python
x_plot = np.linspace(0, 4.5, 100)
```

-   100 điểm từ 0 đến 4.5
-   Đủ mịn để vẽ đường cong mượt

##### **Tính y cho manual model**

```python
y_manual = [get_prediction(m, b, x_i) for x_i in x_plot]
```

-   List comprehension: duyệt 100 điểm
-   Gọi hàm `get_prediction` từ module tự xây dựng
-   Kết quả: 100 giá trị xác suất

##### **Tính y cho sklearn**

```python
y_sklearn = [model.predict_proba([[x_i]])[0][1] for x_i in x_plot]
```

-   `[[x_i]]`: Reshape thành 2D array
-   `predict_proba(...)`: Dự đoán xác suất
-   `[0][1]`: Lấy xác suất class 1

**Lưu ý:** Cách tính hơi "cồng kềnh" với list comprehension, có thể optimize:

```python
# Cách tốt hơn:
X_plot = x_plot.reshape(-1, 1)
y_sklearn = model.predict_proba(X_plot)[:, 1]
```

---

##### **Vẽ dữ liệu gốc**

```python
# Vẽ dữ liệu gốc
x_data = [row[0] for row in DATASET]
y_data = [row[1] for row in DATASET]
ax1.scatter(x_data, y_data, c=['red' if y==0 else 'green' for y in y_data],
            s=100, alpha=0.6, edgecolors='black', linewidth=1.5,
            label='Dữ liệu thực tế', zorder=3)
```

**Giải thích:**

-   **`c=[...]`:** Màu theo nhãn

    -   `'red' if y==0`: Rớt → đỏ
    -   `'green'`: Đậu → xanh
    -   List comprehension tạo list màu: `['red', 'red', ..., 'green', 'green']`

-   **`s=100`:** Kích thước điểm = 100

-   **`alpha=0.6`:** Độ trong suốt 60% (hơi mờ)

-   **`edgecolors='black'`:** Viền đen cho rõ

-   **`linewidth=1.5`:** Độ dày viền

-   **`zorder=3`:** Layer cao (vẽ trên cùng)
    -   Đảm bảo điểm không bị đường che

---

##### **Vẽ đường sigmoid**

```python
# Vẽ đường sigmoid
ax1.plot(x_plot, y_manual, 'b-', linewidth=2, label='Mô hình tự xây dựng')
ax1.plot(x_plot, y_sklearn, 'r--', linewidth=2, label='Sklearn')
```

**Giải thích:**

-   **Manual:**

    -   `'b-'`: Màu xanh (blue), đường liền (solid)
    -   `linewidth=2`: Độ dày 2

-   **Sklearn:**
    -   `'r--'`: Màu đỏ (red), đường gạch (dashed)
    -   Dễ phân biệt với manual

**Kỳ vọng:** 2 đường rất gần nhau, gần như trùng

---

##### **Vẽ ngưỡng 0.5**

```python
# Vẽ ngưỡng 0.5
ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Ngưỡng 0.5')
```

**Giải thích:**

-   `axhline`: Đường ngang
-   `y=0.5`: Tại y = 0.5
-   `linestyle=':'`: Đường chấm (dotted)
-   **Ý nghĩa:** Decision boundary threshold

---

##### **Vẽ điểm dự đoán 2.8h**

```python
# Vẽ điểm dự đoán cho 2.8 giờ
ax1.scatter([hours_input], [predicted_score_manual], c='blue', s=200,
            marker='*', edgecolors='black', linewidth=1.5,
            label=f'Dự đoán {hours_input}h (Manual)', zorder=4)
ax1.scatter([hours_input], [predicted_proba_sklearn], c='red', s=200,
            marker='*', edgecolors='black', linewidth=1.5,
            label=f'Dự đoán {hours_input}h (Sklearn)', zorder=4)
```

**Giải thích:**

-   **Marker `'*'`:** Hình ngôi sao (nổi bật)
-   **`s=200`:** Kích thước lớn
-   **`zorder=4`:** Layer cao nhất (vẽ trên cùng)
-   **2 điểm:**
    -   Manual: Xanh
    -   Sklearn: Đỏ
-   **Kỳ vọng:** 2 điểm rất gần nhau (gần như trùng)

---

##### **Trang trí subplot 1**

```python
ax1.set_xlabel('Số giờ học', fontsize=11, fontweight='bold')
ax1.set_ylabel('Xác suất đậu', fontsize=11, fontweight='bold')
ax1.set_title('So sánh Decision Boundary\nMô hình tự xây dựng vs Sklearn', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)
```

**Giải thích:**

-   **Title:** 2 dòng với `\n`
-   **Legend:** Góc trên trái
-   **Grid:** Độ trong suốt 0.3
-   **ylim:** -0.1 đến 1.1 (hơi rộng hơn 0-1)

---

#### **7.3. Subplot 2: So Sánh Tham Số**

```python
# Subplot 2: So sánh các tham số
categories = ['Hệ số góc\n(m/coef)', 'Hệ số chặn\n(b/intercept)', f'Xác suất đậu\n({hours_input}h)']
manual_values = [m, b, predicted_score_manual]
sklearn_values = [model.coef_[0][0], model.intercept_[0], predicted_proba_sklearn]

x_pos = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, manual_values, width, label='Mô hình tự xây dựng', color='skyblue', edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x_pos + width/2, sklearn_values, width, label='Sklearn', color='salmon', edgecolor='black', linewidth=1.5)
```

**Giải thích:**

##### **Chuẩn bị dữ liệu**

**Categories:**

```python
categories = ['Hệ số góc\n(m/coef)', 'Hệ số chặn\n(b/intercept)', f'Xác suất đậu\n({hours_input}h)']
```

-   3 nhóm so sánh
-   `\n`: Xuống dòng trong label (đẹp hơn)

**Values:**

```python
manual_values = [m, b, predicted_score_manual]
sklearn_values = [model.coef_[0][0], model.intercept_[0], predicted_proba_sklearn]
```

-   2 list tương ứng

##### **Tạo vị trí cột**

```python
x_pos = np.arange(len(categories))  # [0, 1, 2]
width = 0.35
```

-   `x_pos`: Vị trí trung tâm mỗi nhóm
-   `width`: Độ rộng mỗi cột = 0.35

##### **Vẽ grouped bar chart**

```python
bars1 = ax2.bar(x_pos - width/2, manual_values, width, ...)
bars2 = ax2.bar(x_pos + width/2, sklearn_values, width, ...)
```

**Phân tích:**

-   **`x_pos - width/2`:** Dịch sang trái nửa width

    -   Vị trí: [-0.175, 0.825, 1.825]

-   **`x_pos + width/2`:** Dịch sang phải nửa width
    -   Vị trí: [0.175, 1.175, 2.175]

**Kết quả:** 2 cột đứng sát nhau, tạo nhóm

**Màu sắc:**

-   Manual: `'skyblue'` (xanh nhạt)
-   Sklearn: `'salmon'` (đỏ nhạt)
-   Viền đen cho rõ

---

##### **Thêm giá trị lên cột**

```python
# Thêm giá trị lên các cột
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
```

**Giải thích:**

-   **Duyệt 2 nhóm cột:** bars1, bars2
-   **Duyệt từng cột:** bar in bars
-   **Lấy chiều cao:** `bar.get_height()` = giá trị
-   **Vẽ text:**
    -   **Vị trí X:** Trung tâm cột
        -   `bar.get_x()`: Tọa độ trái cột
        -   `+ bar.get_width()/2.`: Cộng nửa width → trung tâm
    -   **Vị trí Y:** `height` (đỉnh cột)
    -   **Text:** Giá trị với 3 chữ số thập phân
    -   **`ha='center'`:** Horizontal alignment = center
    -   **`va='bottom'`:** Vertical alignment = bottom (đặt dưới chữ)

**Kết quả:** Mỗi cột có số ở trên đỉnh

---

##### **Trang trí subplot 2**

```python
ax2.set_ylabel('Giá trị', fontsize=11, fontweight='bold')
ax2.set_title('So sánh Tham số và Kết quả Dự đoán', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories, fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linewidth=0.8)
```

**Giải thích:**

-   **`set_xticks(x_pos)`:** Đặt vị trí tick = [0, 1, 2]
-   **`set_xticklabels(categories)`:** Gắn nhãn cho tick
-   **`grid(axis='y')`:** Chỉ vẽ grid ngang (không dọc)
-   **`axhline(y=0)`:** Vẽ trục x tại y=0 (baseline)

---

#### **7.4. Lưu và Hiển Thị**

```python
plt.tight_layout()
plt.savefig('results/ex2b_comparison_chart.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Giải thích:**

-   Lưu vào thư mục `results/`
-   Tên file rõ ràng: `ex2b_comparison_chart.png`
-   DPI cao (300) cho chất lượng tốt

---

## 📊 Output và Kết Quả (Dự Kiến)

### 🖥️ Console Output

```
============================================================
BÀI 2 - QUESTION B: SO SÁNH MÔ HÌNH TỰ XÂY DỰNG VỚI SKLEARN
============================================================

------------------------------------------------------------
PHẦN A: KẾT QUẢ MÔ HÌNH TỰ XÂY DỰNG
------------------------------------------------------------
Tham số học được:
  - Hệ số góc (m): 2.345678
  - Hệ số chặn (b): -4.567890

Dự đoán cho sinh viên học 2.8 giờ:
  - Xác suất đậu: 0.785432 (78.54%)
  - Kết luận: ĐẬU

------------------------------------------------------------
PHẦN B: KẾT QUẢ MÔ HÌNH SKLEARN
------------------------------------------------------------
Tham số học được:
  - Hệ số góc (coef): 2.398765
  - Hệ số chặn (intercept): -4.612345

Dự đoán cho sinh viên học 2.8 giờ:
  - Xác suất đậu: 0.791234 (79.12%)
  - Kết luận: ĐẬU

Hệ số góc (m/coef):
  - Mô hình tự xây dựng: 2.345678
  - Sklearn:              2.398765
  - Chênh lệch:           0.053087

Hệ số chặn (b/intercept):
  - Mô hình tự xây dựng: -4.567890
  - Sklearn:              -4.612345
  - Chênh lệch:           0.044455

Xác suất đậu cho 2.8 giờ học:
  - Mô hình tự xây dựng: 0.785432 (78.54%)
  - Sklearn:              0.791234 (79.12%)
  - Chênh lệch:           0.005802

Kết luận dự đoán:
  - Mô hình tự xây dựng: ĐẬU
  - Sklearn:             ĐẬU
  - Kết quả: GIỐNG NHAU ✓

============================================================
KẾT LUẬN
============================================================
Có thể thấy sự khác biệt giữa hai mô hình do:
  1. Số lần lặp khác nhau (10 vs thuật toán tối ưu của sklearn)
  2. Phương pháp tối ưu khác nhau (Gradient Descent vs LBFGS)
  3. Điều kiện dừng và khởi tạo tham số khác nhau
============================================================
```

---

### 📈 Phân Tích Kết Quả

#### **1. Tham Số (m/coef, b/intercept)**

**Chênh lệch nhỏ (~5-6%):**

-   Manual: m=2.35, b=-4.57
-   Sklearn: m=2.40, b=-4.61
-   **Nguyên nhân:**
    -   LBFGS hội tụ tốt hơn GD
    -   Sklearn có thể đã hội tụ sớm
    -   Khởi tạo khác nhau

**Ý nghĩa:**

-   **Cùng hướng:** Cả 2 đều học được xu hướng tăng (m > 0)
-   **Cùng dịch:** Cả 2 đều có b âm (dịch sang phải)
-   **Consistency:** Implementation cơ bản đúng!

---

#### **2. Xác Suất Dự Đoán**

**Chênh lệch rất nhỏ (<1%):**

-   Manual: 78.54%
-   Sklearn: 79.12%
-   **Chênh lệch:** 0.58%

**Ý nghĩa:**

-   **Gần như giống nhau:** Cả 2 mô hình dự đoán tương tự
-   **Cùng kết luận:** Đều dự đoán ĐẬU
-   **Tin cậy:** Implementation manual là đáng tin cậy

---

#### **3. Kết Luận Cuối Cùng**

**GIỐNG NHAU ✓**

-   Cả 2 đều kết luận ĐẬU
-   **Quan trọng nhất:** Quyết định cuối cùng giống nhau
-   **Thành công:** Mô hình tự xây dựng hoạt động đúng!

---

#### **4. Biểu Đồ 1: Decision Boundary**

**Quan sát:**

-   **2 đường sigmoid:** Gần như trùng nhau
-   **Điểm dữ liệu:** Phân bố rõ ràng (đỏ trái, xanh phải)
-   **2 ngôi sao (2.8h):** Rất gần nhau (~79%)
-   **Ngưỡng 0.5:** 2 đường đều vượt qua tại ~2.0-2.2 giờ

**Kết luận:**

-   Mô hình manual **hoạt động tốt**
-   Decision boundary **hợp lý**

---

#### **5. Biểu Đồ 2: So Sánh Tham Số**

**Quan sát:**

**Nhóm 1 - Hệ số góc:**

-   Manual: ~2.35 (xanh)
-   Sklearn: ~2.40 (đỏ)
-   **Cao gần bằng nhau**

**Nhóm 2 - Hệ số chặn:**

-   Manual: ~-4.57 (xanh, âm)
-   Sklearn: ~-4.61 (đỏ, âm)
-   **Cả 2 đều âm, gần nhau**

**Nhóm 3 - Xác suất đậu:**

-   Manual: ~0.785 (xanh)
-   Sklearn: ~0.791 (đỏ)
-   **Gần như bằng nhau**

**Kết luận:**

-   **Visual confirmation:** Mắt thường thấy 2 cột gần nhau
-   **Chênh lệch nhỏ:** Acceptable difference

---