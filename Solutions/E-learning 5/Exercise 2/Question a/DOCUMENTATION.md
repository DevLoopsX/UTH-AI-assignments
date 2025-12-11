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

Đoạn code bắt đầu bằng việc import hai module hệ thống quan trọng. Module `sys` cung cấp các chức năng để thao tác với môi trường runtime của Python, trong khi module `os` cho phép làm việc với hệ điều hành, đặc biệt là các thao tác liên quan đến file và thư mục.

#### **Thêm đường dẫn module**

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

**Phân tích từng bước:**

Quá trình xử lý đường dẫn diễn ra qua năm bước tuần tự. Bước đầu tiên, biến đặc biệt `__file__` chứa đường dẫn của file Python hiện tại, ví dụ như `d:/UTH-AI-assignments/.../Exercise 2/Question a/ex2a_logistic_regression.py`. Tiếp theo, hàm `os.path.abspath(__file__)` chuyển đổi đường dẫn này thành dạng tuyệt đối, đảm bảo đường dẫn đầy đủ và không phụ thuộc vào thư mục làm việc hiện tại.

Bước thứ ba áp dụng `os.path.dirname(...)` lần đầu để lấy thư mục cha, cho kết quả `d:/UTH-AI-assignments/.../Exercise 2/Question a/`. Sau đó, áp dụng `os.path.dirname(...)` lần thứ hai để lên thêm một cấp nữa, thu được `d:/UTH-AI-assignments/.../Exercise 2/`. Cuối cùng, `sys.path.insert(0, ...)` thêm đường dẫn này vào vị trí đầu tiên (index 0) của danh sách tìm kiếm module, tạo ưu tiên cao nhất. Điều này cho phép Python tìm thấy file `logistic_regression_utils.py` nằm ở thư mục cha.

**Tại sao cần làm thế này?**

Lý do cần thao tác này xuất phát từ cấu trúc thư mục của project. File `logistic_regression_utils.py` nằm ở thư mục cha `Exercise 2/`, trong khi file hiện tại nằm ở thư mục con `Exercise 2/Question a/`. Do Python mặc định chỉ tìm kiếm module trong thư mục hiện tại và các thư mục trong `sys.path`, nên cần phải thêm đường dẫn thủ công để có thể import từ thư mục cha.

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

Đoạn import này lấy ba thành phần quan trọng từ module tiện ích. Biến `DATASET` là hằng số chứa toàn bộ dữ liệu training gồm 8 cặp giá trị (hours, pass). Hàm `get_prediction` thực hiện dự đoán xác suất cho các giá trị input mới dựa trên tham số đã học. Còn hàm `train_logistic_regression` chịu trách nhiệm huấn luyện mô hình với thuật toán Gradient Descent.

**Lợi ích của cách tổ chức này:**

Việc tổ chức code theo cách này mang lại nhiều ưu điểm quan trọng. Thứ nhất, các hàm được tái sử dụng cho cả Question A và Question B, tránh việc viết lại code trùng lặp. Thứ hai, khi cần sửa đổi logic, chỉ cần thay đổi ở một chỗ trong file utils, và cả hai file sử dụng đều được cập nhật tự động. Cuối cùng, file chính trở nên sạch sẽ và tập trung vào logic cụ thể của từng câu hỏi.

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

Sau khi huấn luyện xong, biến `hours_input` được gán giá trị 2.8, thể hiện trường hợp sinh viên học 2.8 giờ theo yêu cầu của đề bài. Tiếp đó, hàm `get_prediction(m, b, hours_input)` được gọi để thực hiện dự đoán. Hàm này nhận ba tham số đầu vào: các tham số m và b vừa học được từ quá trình training, cùng với giá trị x cần dự đoán (2.8 giờ). Kết quả trả về là xác suất đậu nằm trong khoảng từ 0 đến 1.

**Công thức bên trong hàm:**

Quá trình tính toán trong hàm `get_prediction` diễn ra qua hai bước. Bước đầu tiên tính giá trị tuyến tính:

$$z = m \times x + b$$

Sau đó áp dụng hàm sigmoid để chuyển đổi z thành xác suất:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Với các tham số m, b đã học và x = 2.8, ví dụ nếu m=2.0 và b=-4.0, quá trình tính toán như sau:

$$z = 2.0 \times 2.8 + (-4.0) = 5.6 - 4.0 = 1.6$$

$$\sigma(1.6) = \frac{1}{1 + e^{-1.6}} \approx 0.832$$

Kết quả này nghĩa là xác suất đậu xấp xỉ 83.2%.

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

Lệnh này tạo một dòng phân cách bằng cách xuống dòng (`"\n"`) và sau đó in 40 dấu gạch ngang. Kết quả là một separator đẹp mắt giúp tách biệt phần kết quả khỏi các phần khác.

#### **In kết quả dự đoán**

Phần này hiển thị kết quả dự đoán dưới nhiều dạng khác nhau. Lệnh `print(f"Kết quả dự đoán cho {hours_input} giờ học:")` in ra số giờ input (2.8) để người dùng biết đang dự đoán cho trường hợp nào. Tiếp theo, `print(f"Điểm số dự đoán: {predicted_score:.4f}")` in xác suất với 4 chữ số thập phân (ví dụ 0.8324). Cuối cùng, lệnh `print(f"Xác suất đậu: {predicted_score:.4f} ({predicted_score*100:.2f}%)")` hiển thị cả dạng thập phân (0.8324) và dạng phần trăm (83.24%), trong đó `.2f` format số phần trăm với 2 chữ số thập phân.

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

Ngưỡng quyết định có thể được điều chỉnh tùy theo mục đích sử dụng. Nếu cần "cẩn thận hơn", có thể đặt threshold = 0.7 để chỉ kết luận đậu khi rất chắc chắn. Ngược lại, nếu muốn "dễ dãi hơn", có thể đặt threshold = 0.3 để dễ dàng kết luận đậu hơn. Tuy nhiên, việc điều chỉnh này tạo ra trade-off giữa Precision và Recall cần cân nhắc.

#### **Kết thúc**

```python
print("-"*40 + "\n")
```

Dòng code này in separator đóng và xuống dòng để kết quả output trông thoáng mắt hơn.

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

Đây là một danh sách 2D trong đó mỗi phần tử là `[hours, pass]` đại diện cho một mẫu dữ liệu. Dataset chứa 8 mẫu với 4 mẫu rớt (từ 0.5 đến 2.0 giờ với y=0) và 4 mẫu đậu (từ 2.5 đến 4.0 giờ với y=1).

**Phân tích dataset:**

Dataset này có dữ liệu cân bằng với 50% đậu và 50% rớt. Đặc điểm nổi bật là phân chia rõ ràng với khoảng trống giữa 2.0 và 2.5 giờ, do đó có thể vẽ đường phân chia khá tuyến tính. Decision boundary dự kiến sẽ nằm khoảng 2.2-2.3 giờ.

---

### **2. Hàm get_prediction**

```python
def get_prediction(m, b, x):
    # Sigmoid function
    y = m * x + b
    return 1 / (1 + math.exp(-y))
```

**Giải thích:**

Hàm này thực hiện **forward propagation** (truyền tiến) qua hai bước chính. Đầu tiên, tính giá trị tuyến tính:

```python
y = m * x + b
```

Biến `y` ở đây thực ra là `z` (pre-activation), đại diện cho phương trình đường thẳng:

$$z = mx + b$$

Tiếp theo, áp dụng sigmoid để chuyển đổi:

```python
return 1 / (1 + math.exp(-y))
```

Công thức sigmoid:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Hàm này chuyển giá trị z thành xác suất trong khoảng (0, 1).

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

Hàm tính **Binary Cross-Entropy Loss** tương tự Exercise 1 nhưng với cách implement khác.

#### **Tham số:**

Tham số **`y`** là list các nhãn thực tế như `[0, 0, 0, 0, 1, 1, 1, 1]`, trong khi **`y_hat`** là list các xác suất dự đoán như `[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]`.

#### **Cách tính:**

Đầu tiên, đếm số mẫu:

```python
k = len(y)
```

Biến k sẽ bằng 8 (số mẫu trong dataset). Tiếp theo, khởi tạo tổng cost:

```python
total_cost = 0.0
```

Sau đó duyệt từng cặp (yi, y_hat_i):

```python
for yi, y_hat_i in zip(y, y_hat):
```

Hàm `zip(y, y_hat)` ghép từng cặp tương ứng, ví dụ (0, 0.1), (0, 0.2), ..., (1, 0.9). Trong mỗi vòng lặp, cộng dồn loss:

```python
total_cost += -(yi * math.log(y_hat_i) + (1 - yi) * math.log(1 - y_hat_i))
```

Nếu yi = 1 thì chỉ tính `-log(y_hat_i)`, còn nếu yi = 0 thì chỉ tính `-log(1 - y_hat_i)`. Cuối cùng, lấy trung bình:

```python
return total_cost / k
```

Kết quả được chia cho k để lấy giá trị trung bình.

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

Hàm tính **gradient** (đạo hàm) của Cost function tương tự Exercise 1.

#### **Tham số:**

Các tham số **`m, b`** là tham số hiện tại (không dùng trong hàm này, chỉ để tương thích signature). Tham số **`x`** là list giá trị features như `[0.5, 1.0, 1.5, ..., 4.0]`, **`y`** là list nhãn thực tế `[0, 0, 0, ..., 1]`, và **`y_hat`** là list xác suất dự đoán `[h1, h2, ..., h8]`.

#### **Cách tính:**

Đầu tiên, tính gradient của m (hệ số góc):

```python
dm = (1 / k) * sum((y_hat_i - yi) * xi for y_hat_i, yi, xi in zip(y_hat, y, x))
```

Công thức toán học:

$$\frac{\partial J}{\partial m} = \frac{1}{k}\sum_{i=1}^{k}(h_i - y_i) \cdot x_i$$

Trong công thức, `(y_hat_i - yi)` là sai số tại điểm thứ i, sau đó nhân với `xi` để tính weighted error. Hàm `sum(...)` tính tổng trên tất cả điểm, rồi nhân với `(1 / k)` để lấy trung bình. Generator expression `for y_hat_i, yi, xi in zip(y_hat, y, x)` duyệt qua 3 list cùng lúc một cách Pythonic và gọn hơn vòng for thông thường.

Tiếp theo, tính gradient của b (hệ số chặn):

```python
db = (1 / k) * sum(y_hat_i - yi for y_hat_i, yi in zip(y_hat, y))
```

Công thức toán học:

$$\frac{\partial J}{\partial b} = \frac{1}{k}\sum_{i=1}^{k}(h_i - y_i)$$

Cách tính tương tự dm nhưng không nhân với xi, vì đạo hàm của b là 1.

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

Đầu tiên, chuyển xác suất thành nhãn:

```python
1 if y_hat_i >= 0.5 else 0
```

Nếu y_hat_i ≥ 0.5 thì dự đoán 1 (ĐẬU), ngược lại dự đoán 0 (RỚT). Tiếp theo, so sánh với nhãn thực:

```python
(... == yi)
```

Kết quả trả về True nếu dự đoán đúng, False nếu dự đoán sai. Sau đó đếm số dự đoán đúng:

```python
correct_predictions = sum(...)
```

Hàm `sum` trên boolean cho True=1 và False=0, do đó kết quả là tổng số dự đoán đúng. Cuối cùng, tính tỷ lệ:

```python
return correct_predictions / len(y)
```

Kết quả là số đúng chia cho tổng số mẫu, cho giá trị từ 0.0 (0%) đến 1.0 (100%).

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

Đây là **hàm chính** thực hiện thuật toán Gradient Descent, đóng vai trò trái tim của bài toán.

#### **Tham số:**

Hàm nhận tham số **`dataset`** là dữ liệu training (mặc định DATASET), **`m_init, b_init`** là giá trị khởi tạo (mặc định 1.0, -1.0), **`iterations`** là số vòng lặp (mặc định 10), và **`learning_rate`** là tốc độ học (mặc định 1.0).

#### **Bước 1: Khởi tạo tham số**

```python
m = m_init
b = b_init
```

Đoạn code này gán giá trị ban đầu cho m và b.

#### **Bước 2: Tách dữ liệu**

```python
x = [row[0] for row in dataset]
y = [row[1] for row in dataset]
```

Sử dụng list comprehension để trích xuất dữ liệu. Biểu thức `row[0]` lấy cột đầu tiên (hours), còn `row[1]` lấy cột thứ hai (pass). Kết quả:

```python
x = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
y = [0, 0, 0, 0, 1, 1, 1, 1]
```

#### **Bước 3: Khởi tạo list lưu cost**

```python
costs = []
```

List này phục vụ cho việc tracking quá trình học.

#### **Bước 4: Vòng lặp training**

```python
for it in range(iterations):
```

Vòng lặp này thực hiện `iterations` lần (10 lần).

**Bên trong mỗi iteration:**

Đầu tiên là **4.1. Forward propagation:**

```python
y_hat = [get_prediction(m, b, xi) for xi in x]
```

Bước này tính xác suất dự đoán cho từng điểm, kết quả `y_hat` là list 8 giá trị xác suất. Tiếp theo, **4.2. Tính cost:**

```python
cost = get_cost(y, y_hat)
costs.append(cost)
```

Đoạn code này đánh giá chất lượng mô hình hiện tại và lưu vào list. Sau đó thực hiện **4.3. Backward propagation (tính gradient):**

```python
dm, db = get_gradients(m, b, x, y, y_hat)
```

Bước này tính đạo hàm để biết hướng đi. Cuối cùng là **4.4. Cập nhật tham số:**

```python
m -= learning_rate * dm
b -= learning_rate * db
```

Đây là Gradient Descent step, đi ngược hướng gradient để giảm cost.

#### **Bước 5: Trả về kết quả**

```python
return m, b, costs
```

Hàm trả về `m, b` là tham số tối ưu sau training, và `costs` là lịch sử cost để phân tích.

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

Sinh viên học 2.8 giờ có xác suất đậu gần **80%**. Kết quả không phải 100% vì mô hình học từ dữ liệu có **uncertainty**. Dự đoán này là hợp lý do 2.8 giờ lớn hơn 2.5 giờ (điểm đậu thấp nhất) và gần với 3.0 giờ (điểm đậu chắc chắn).

#### **2. Kết luận: ĐẬU**

**Logic:**

Xác suất 0.7854 ≥ 0.5 nên được phân loại vào lớp 1 (ĐẬU). Mức độ tin cậy là **cao** vì xác suất gần 80%, không phải chỉ 51%.

#### **3. Phân tích theo ngưỡng giờ học**

Giả sử mô hình học được m ≈ 2.0 và b ≈ -4.0. Decision boundary được tính khi m × x + b = 0, do đó x = -b/m = 4.0/2.0 = 2.0 giờ.

**Dự đoán theo ngưỡng:**

Nếu x < 2.0 giờ thì xác suất đậu < 50% nên kết luận RỚT. Nếu x = 2.0 giờ thì xác suất đậu = 50% (biên giới). Nếu x > 2.0 giờ thì xác suất đậu > 50% nên kết luận ĐẬU. Vì **2.8 giờ > 2.0 giờ** nên kết luận ĐẬU là hợp lý!

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

Mô hình dự đoán hợp lý với dữ liệu thực tế. Giá trị 2.8 giờ nằm giữa 2.5 giờ (73%) và 3.0 giờ (88%), do đó kết quả ~79% là hoàn toàn hợp lý.

---
