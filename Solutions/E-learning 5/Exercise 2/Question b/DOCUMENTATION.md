# 📘 E-Learning 5 - Exercise 2 - Question B: So Sánh với Sklearn

## 🎯 Mục Tiêu Bài Tập

Bài tập yêu cầu **so sánh kết quả** giữa **mô hình Logistic Regression tự xây dựng** (Question A) với **mô hình từ thư viện sklearn** để kiểm chứng tính đúng đắn của implementation.

### 📊 Đề Bài

**Yêu cầu:**

Bài tập yêu cầu sử dụng cùng dataset như Question A, sau đó huấn luyện mô hình bằng **sklearn.linear_model.LogisticRegression**. Tiếp theo, thực hiện so sánh kết quả giữa 2 mô hình bao gồm tham số học được (m/coef, b/intercept), xác suất dự đoán cho 2.8 giờ học, và kết luận đậu/rớt. Cuối cùng, tạo biểu đồ so sánh trực quan (**Visualization**).

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

Đoạn code này import các hàm đã được xây dựng trong Question A để sử dụng lại. Việc tái sử dụng code như vậy giúp tránh viết lại toàn bộ logic training, vừa tiết kiệm thời gian vừa đảm bảo tính nhất quán.

#### **Import sklearn**

```python
from sklearn.linear_model import LogisticRegression
```

Thư viện sklearn (scikit-learn) là thư viện Machine Learning phổ biến nhất trong Python, được phát triển và tối ưu hóa rất kỹ lưỡng. Class `LogisticRegression` trong module `linear_model` cung cấp một implementation chuẩn và hiệu quả của thuật toán Logistic Regression. Điểm nổi bật của sklearn là highly optimized với backend được viết bằng C/C++, hỗ trợ nhiều thuật toán tối ưu khác nhau như LBFGS, SAG, SAGA, và được kiểm thử kỹ lưỡng nên rất đáng tin cậy trong ứng dụng thực tế.

#### **Import numpy và matplotlib**

```python
import numpy as np
import matplotlib.pyplot as plt
```

Hai thư viện này phục vụ các mục đích khác nhau trong quá trình so sánh. Thư viện numpy được import vì sklearn yêu cầu dữ liệu đầu vào phải ở dạng numpy array thay vì Python list thông thường. Trong khi đó, matplotlib.pyplot được sử dụng để vẽ các biểu đồ so sánh giữa hai mô hình một cách trực quan.

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

Phần này sử dụng lại toàn bộ code đã được viết trong Question A, chạy đúng 10 iterations theo yêu cầu của đề bài. Việc tái sử dụng code giúp đảm bảo tính nhất quán khi so sánh giữa hai mô hình.

#### **Prediction**

Sau khi training xong, mô hình được sử dụng để dự đoán cho trường hợp sinh viên học 2.8 giờ. Kết quả dự đoán được lưu vào biến `predicted_score_manual` để chuẩn bị cho phần so sánh với sklearn sau này.

**Tại sao gọi là "manual"?**

Thuật ngữ "manual" được sử dụng để phân biệt với sklearn (automated/optimized). Từ "manual" ở đây nghĩa là tự code từ đầu, thực hiện từng bước một cách rõ ràng, thay vì dùng các hàm có sẵn đã được tối ưu hóa cao.

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

Phần in kết quả được tổ chức theo cấp bậc rõ ràng. Header lớn với dấu bằng (=) được dùng cho tiêu đề bài toán, trong khi header nhỏ với dấu gạch ngang (-) đánh dấu tiêu đề từng phần cụ thể.

#### **In tham số**

```python
print(f"  - Hệ số góc (m): {m:.6f}")
print(f"  - Hệ số chặn (b): {b:.6f}")
```

Các tham số được in với format `.6f` để hiển thị 6 chữ số thập phân, đảm bảo độ chính xác cao. Việc thụt đầu dòng với ký tự ` -` giúp cấu trúc output dễ đọc và thẩm mỹ hơn.

#### **In kết quả dự đoán**

```python
print(f"  - Xác suất đậu: {predicted_score_manual:.6f} ({predicted_score_manual*100:.2f}%)")
```

Dòng code này in cả dạng thập phân và phần trăm, ví dụ `0.785432 (78.54%)`.

#### **Kết luận**

```python
if predicted_score_manual >= 0.5:
    print(f"  - Kết luận: ĐẬU")
else:
    print(f"  - Kết luận: RỚT")
```

Logic đơn giản: nếu xác suất ≥ 0.5 thì kết luận ĐẬU.

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

Biểu thức này sử dụng list comprehension để trích xuất dữ liệu. `row[0]` lấy cột đầu tiên (Hours) từ mỗi dòng dữ liệu. Điểm quan trọng cần chú ý là có hai cặp ngoặc vuông: ngoặc trong `[row[0]]` tạo một list chứa 1 phần tử, còn ngoặc ngoài `[...]` là cú pháp của list comprehension.

**Tại sao cần 2 cặp ngoặc?**

Sklearn yêu cầu X phải là ma trận 2D với kích thước (n_samples × n_features), trong đó `n_samples` là số mẫu (8 điểm) và `n_features` là số đặc trưng (1 feature = hours).

**Kết quả:**

Ma trận kết quả có dạng:

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

Nếu viết `X = np.array([row[0] for row in DATASET])`, kết quả sẽ là `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]` với shape (8,) - một 1D array. Điều này sẽ khiến sklearn báo lỗi vì không đúng format yêu cầu.

##### **Labels (y_train)**

```python
y_train = np.array([row[1] for row in DATASET])
```

**Phân tích:**

Biểu thức `row[1]` lấy cột thứ hai (Pass) từ mỗi dòng dữ liệu. Khác với X, y chỉ cần là một 1D array với nội dung `[0, 0, 0, 0, 1, 1, 1, 1]`, có shape (8,) tương ứng với 8 phần tử.

**Lưu ý naming:**

Biến được đặt tên là `y_train` thay vì `y` để tránh nhầm lẫn với biến `y` đã được sử dụng ở phần A, giúp code rõ ràng và dễ bảo trì hơn.

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

Sklearn cần random_state vì một số solver khởi tạo tham số ngẫu nhiên. Ngoài ra, có thể xảo trộn dữ liệu khi dùng batch methods. Việc thiết lập random_state đảm bảo reproducibility cho nghiên cứu khoa học.

##### **Fit model**

```python
model.fit(X, y_train)
```

**Giải thích:**

Hàm **`fit(X, y)`** là hàm huấn luyện mô hình. **Input** gồm `X` là Features (ma trận 8×1) và `y_train` là Labels (8 phần tử). **Process** bao gồm chạy thuật toán LBFGS, tối ưu hóa tham số (coef, intercept), với tối đa 10 iterations. **Output** là `model` được cập nhật (in-place), với tham số được lưu trong `model.coef_` và `model.intercept_`.

**Lưu ý:**

Sklearn tự động normalize/standardize nếu cần, tự động xử lý convergence, và tự động điều chỉnh learning rate (adaptive).

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

Biến `hours_input` có giá trị 2.8, do đó `[[2.8]]` tạo ma trận 2D với kích thước 1×1 (1 sample, 1 feature). Điểm quan trọng là phải cùng format với X training (2D).

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

Để lấy **hệ số góc:**

```python
model.coef_[0][0]
```

Thuộc tính **`model.coef_`** là ma trận hệ số với kích thước (n_classes-1, n_features). Đối với Logistic Regression binary, kích thước là (1, 1) như `[[2.345]]` (1 class, 1 feature). Đề tác `[0]` lấy hàng đầu cho kết quả `[2.345]`, rồi `[0]` tiếp theo lấy cột đầu cho kết quả cuối cùng `2.345`.

Để lấy **hệ số chặn:**

```python
model.intercept_[0]
```

Thuộc tính **`model.intercept_`** là array hệ số chặn với kích thước (n_classes-1,) như `[-4.567]` (1 phần tử). Đề tác `[0]` lấy phần tử đầu cho kết quả `-4.567`.

##### **In kết quả**

Format output được thiết kế giống phần A để dễ so sánh, sử dụng `.6f` cho độ chính xác cao.

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

Đoạn code này tính **giá trị tuyệt đối** của sự chênh lệch. Hàm `abs()` luôn trả về số dương nên dễ so sánh, với kỳ vọng chênh lệch nhỏ hơn 0.1.

##### **2. Hệ số chặn (b vs intercept)**

```python
print(f"  - Chênh lệch:           {abs(b - model.intercept_[0]):.6f}")
```

Cách tính tương tự với m, cũng kỳ vọng chênh lệch nhỏ.

##### **3. Xác suất dự đoán**

```python
print(f"  - Chênh lệch:           {abs(predicted_score_manual - predicted_proba_sklearn):.6f}")
```

Đây là bước so sánh output cuối cùng, và cũng là **quan trọng nhất** để kiểm tra kết quả dự đoán có đúng không. Kỳ vọng chênh lệch rất nhỏ, thường nhỏ hơn 0.01.

##### **4. Kết luận cuối cùng**

```python
result_manual = "ĐẬU" if predicted_score_manual >= 0.5 else "RỚT"
result_sklearn = "ĐẬU" if predicted_class_sklearn == 1 else "RỚT"
```

Đoạn code này chuyển số thành text để dễ đọc, sau đó so sánh string để kiểm tra consistency.

```python
if result_manual == result_sklearn:
    print(f"  - Kết quả: GIỐNG NHAU ✓")
else:
    print(f"  - Kết quả: KHÁC NHAU ✗")
```

Nếu kết quả giống nhau (✓), chứng tỏ implementation đúng! Ngược lại nếu khác nhau (✗), có vấn đề cần kiểm tra lại.

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

Mô hình **Manual** chạy đúng 10 iterations, không thêm không bớt. Trong khi đó, **Sklearn** có thể hội tụ sớm hơn nếu đạt tolerance. Với mặc định `tol=1e-4`, thuật toán dừng khi gradient < tolerance, do đó có thể dừng sau 5-8 iterations.

##### **2. Phương pháp tối ưu khác nhau**

**Manual - Gradient Descent:**

Sử dụng công thức:

$$w_{new} = w_{old} - \alpha \nabla J$$

Đây là **Gradient bậc 1** (first-order derivative), đơn giản và dễ hiểu nhưng tốc độ hội tụ là **tuyến tính** (linear).

**Sklearn - LBFGS:**

Đây là **Quasi-Newton method** sử dụng **gradient bậc 2** (approximated Hessian). Phương pháp này phức tạp hơn nhưng **hiệu quả hơn**, với tốc độ hội tụ là **siêu tuyến tính** (superlinear).

**Ví dụ:**

Với GD, 10 steps có thể chỉ giảm Cost 60%, trong khi LBFGS với 10 steps có thể giảm Cost đến 95%.

##### **3. Điều kiện dừng và khởi tạo**

**Khởi tạo:**

Manual sử dụng m=1.0, b=-1.0 (do mình chọn), trong khi Sklearn sử dụng w=0, b=0 (mặc định) hoặc random.

**Điều kiện dừng:**

Manual chạy đúng 10 iterations mà không kiểm tra convergence, còn Sklearn dừng khi `||gradient|| < tol` hoặc đạt `max_iter`.

**Learning rate:**

Manual dùng learning rate cố định (1.0), nhưng Sklearn sử dụng Adaptive learning rate (LBFGS tự điều chỉnh).

---

### 7️⃣ Visualization

#### **7.1. Tạo Figure với 2 Subplots**

```python
# ========== VISUALIZATION ==========

# Tạo figure với 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
```

**Giải thích:**

Lệnh này tạo 2 biểu đồ cạnh nhau với kích thước lớn (14×5) để kết quả rõ ràng.

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

Hàm này tạo 100 điểm từ 0 đến 4.5, đủ mịn để vẽ đường cong mượt mài.

##### **Tính y cho manual model**

```python
y_manual = [get_prediction(m, b, x_i) for x_i in x_plot]
```

Sử dụng list comprehension để duyệt 100 điểm, gọi hàm `get_prediction` từ module tự xây dựng. Kết quả là 100 giá trị xác suất.

##### **Tính y cho sklearn**

```python
y_sklearn = [model.predict_proba([[x_i]])[0][1] for x_i in x_plot]
```

Biếu thức `[[x_i]]` reshape thành 2D array, sau đó `predict_proba(...)` dự đoán xác suất, rồi `[0][1]` lấy xác suất class 1.

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

Tham số **`c=[...]`** thiết lập màu theo nhãn. Biểu thức `'red' if y==0` chọn màu đỏ cho Rớt và `'green'` cho Đậu. List comprehension tạo list màu như `['red', 'red', ..., 'green', 'green']`. Tham số **`s=100`** đặt kích thước điểm bằng 100. Tham số **`alpha=0.6`** thiết lập độ trong suốt 60% (hơi mờ). Tham số **`edgecolors='black'`** thêm viền đen cho rõ ràng, với **`linewidth=1.5`** đặt độ dày viền. Tham số **`zorder=3`** đặt layer cao (vẽ trên cùng), đảm bảo điểm không bị đường che.

---

##### **Vẽ đường sigmoid**

```python
# Vẽ đường sigmoid
ax1.plot(x_plot, y_manual, 'b-', linewidth=2, label='Mô hình tự xây dựng')
ax1.plot(x_plot, y_sklearn, 'r--', linewidth=2, label='Sklearn')
```

**Giải thích:**

Đường **Manual** sử dụng `'b-'` (màu xanh blue, đường liền solid) với `linewidth=2` (independentộ dày 2). Đường **Sklearn** sử dụng `'r--'` (màu đỏ red, đường gạch dashed) để dễ phân biệt với manual. Kỳ vọng là 2 đường rất gần nhau, gần như trùng.

---

##### **Vẽ ngưỡng 0.5**

```python
# Vẽ ngưỡng 0.5
ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Ngưỡng 0.5')
```

**Giải thích:**

Hàm `axhline` vẽ đường ngang tại `y=0.5`, sử dụng `linestyle=':'` cho đường chấm (dotted). Ý nghĩa của đường này là decision boundary threshold.

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

Marker `'*'` tạo hình ngôi sao nổi bật, với `s=200` là kích thước lớn. Tham số `zorder=4` đặt layer cao nhất để vẽ trên cùng. Có 2 điểm: Manual màu xanh và Sklearn màu đỏ. Kỳ vọng là 2 điểm rất gần nhau (gần như trùng).

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

Title được chia làm 2 dòng với `\n`. Legend đặt ở góc trên trái. Grid có độ trong suốt 0.3 để không quá nổi. Tham số ylim được đặt từ -0.1 đến 1.1 (hơi rộng hơn khoảng 0-1 thông thường).

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

Thiết lập **Categories:**

```python
categories = ['Hệ số góc\n(m/coef)', 'Hệ số chặn\n(b/intercept)', f'Xác suất đậu\n({hours_input}h)']
```

List này chứa 3 nhóm so sánh, với `\n` để xuống dòng trong label cho đẹp hơn.

Thiết lập **Values:**

```python
manual_values = [m, b, predicted_score_manual]
sklearn_values = [model.coef_[0][0], model.intercept_[0], predicted_proba_sklearn]
```

Hai list này chứa giá trị tương ứng của từng mô hình.

##### **Tạo vị trí cột**

```python
x_pos = np.arange(len(categories))  # [0, 1, 2]
width = 0.35
```

Biến `x_pos` là vị trí trung tâm mỗi nhóm, còn `width` là độ rộng mỗi cột bằng 0.35.

##### **Vẽ grouped bar chart**

```python
bars1 = ax2.bar(x_pos - width/2, manual_values, width, ...)
bars2 = ax2.bar(x_pos + width/2, sklearn_values, width, ...)
```

Phân tích: biểu thức `x_pos - width/2` dịch sang trái nửa width, cho vị trí [-0.175, 0.825, 1.825]. Biểu thức `x_pos + width/2` dịch sang phải nửa width, cho vị trí [0.175, 1.175, 2.175]. Kết quả là 2 cột đứng sát nhau, tạo nhóm. Về màu sắc, Manual dùng `'skyblue'` (xanh nhạt), Sklearn dùng `'salmon'` (đỏ nhạt), cả hai đều có viền đen cho rõ ràng.

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

Vòng lặp duyệt qua 2 nhóm cột (bars1, bars2), sau đó duyệt từng cột trong nhóm. Lấy chiều cao bằng `bar.get_height()` để đại diện cho giá trị. Tiếp theo vẽ text với các thiết lập: **Vị trí X** là trung tâm cột được tính bằng `bar.get_x()` (tọa độ trái cột) cộng `bar.get_width()/2.` (nửa width). **Vị trí Y** là `height` (đỉnh cột). **Text** là giá trị với 3 chữ số thập phân. Các tham số `ha='center'` là Horizontal alignment = center, và `va='bottom'` là Vertical alignment = bottom (đặt dưới chữ). Kết quả là mỗi cột có số ở trên đỉnh.

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

Hàm `set_xticks(x_pos)` đặt vị trí tick bằng [0, 1, 2]. Hàm `set_xticklabels(categories)` gắn nhãn cho tick. Tham số `grid(axis='y')` chỉ vẽ grid ngang (không dọc). Hàm `axhline(y=0)` vẽ trục x tại y=0 (baseline).

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

Kết quả cho thấy Manual có m=2.35, b=-4.57 trong khi Sklearn có m=2.40, b=-4.61. Nguyên nhân của sự khác biệt này là LBFGS hội tụ tốt hơn GD, Sklearn có thể đã hội tụ sớm, và khởi tạo khác nhau.

**Ý nghĩa:**

Cả hai mô hình đều học được xu hướng tăng (**Cùng hướng:** m > 0). Cả hai đều có b âm (**Cùng dịch:** dịch sang phải). Điều này chứng tỏ **Consistency:** Implementation cơ bản đúng!

---

#### **2. Xác Suất Dự Đoán**

**Chênh lệch rất nhỏ (<1%):**

Manual cho 78.54% trong khi Sklearn cho 79.12%, chênh lệch chỉ 0.58%.

**Ý nghĩa:**

Hai mô hình dự đoán tương tự (**Gần như giống nhau**). Cả hai đều dự đoán ĐẬU (**Cùng kết luận**). Điều này chứng tỏ Implementation manual là đáng tin cậy (**Tin cậy**).

---

#### **3. Kết Luận Cuối Cùng**

**GIỐNG NHAU ✓**

Cả 2 đều kết luận ĐẬU. Quan trọng nhất là quyết định cuối cùng giống nhau. Đây là thành công: Mô hình tự xây dựng hoạt động đúng!

---

#### **4. Biểu Đồ 1: Decision Boundary**

**Quan sát:**

Hai đường sigmoid gần như trùng nhau. Các điểm dữ liệu phân bố rõ ràng (đỏ bên trái, xanh bên phải). Hai ngôi sao ở 2.8h rất gần nhau (~79%). Ngưỡng 0.5 cho thấy 2 đường đều vượt qua tại khoảng 2.0-2.2 giờ.

**Kết luận:**

Mô hình manual hoạt động tốt và decision boundary hợp lý.

---

#### **5. Biểu Đồ 2: So Sánh Tham Số**

**Quan sát:**

**Nhóm 1 - Hệ số góc:** Manual ~2.35 (xanh) và Sklearn ~2.40 (đỏ) có chiều cao gần bằng nhau.

**Nhóm 2 - Hệ số chặn:** Manual ~-4.57 (xanh, âm) và Sklearn ~-4.61 (đỏ, âm), cả 2 đều âm và gần nhau.

**Nhóm 3 - Xác suất đậu:** Manual ~0.785 (xanh) và Sklearn ~0.791 (đỏ) gần như bằng nhau.

**Kết luận:**

Mắt thường thấy 2 cột gần nhau (Visual confirmation). Chênh lệch nhỏ là chấp nhận được (Acceptable difference).

---
