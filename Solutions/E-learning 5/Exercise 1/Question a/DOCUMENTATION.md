# 📘 E-Learning 5 - Exercise 1 - Question A: Tính Hàm Chi Phí J(w,b)

## 🎯 Mục Tiêu Bài Tập

Bài tập yêu cầu tính toán giá trị của **hàm chi phí (Cost Function)** J(w,b) cho bài toán **Logistic Regression** với các tham số ban đầu được cho trước.

### 📊 Đề Bài

Cho tập dữ liệu như sau:

| x   | y   |
| --- | --- |
| 0.5 | 0   |
| 1.0 | 0   |
| 1.5 | 0   |
| 3.0 | 1   |
| 2.0 | 1   |
| 1.0 | 1   |

**Thông số ban đầu:**

-   w (trọng số) = 0
-   b (bias) = 0
-   α (learning rate) = 0.0001

**Yêu cầu:** Tính J(w,b) - Hàm chi phí Binary Cross-Entropy

---

## 💻 Phân Tích Source Code Chi Tiết

### 1️⃣ Import Thư Viện và Khởi Tạo Dữ Liệu

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.5, 1, 1.5, 3, 2, 1])
y = np.array([0, 0, 0, 1, 1, 1])

# Khởi tạo tham số ban đầu cho thuật toán
w = 0
b = 0
alpha = 0.0001
```

**Giải thích:**

Đoạn code này thực hiện việc **chuẩn bị môi trường** và **khởi tạo dữ liệu** cho bài toán Logistic Regression:

-   **`numpy`**: Thư viện toán học mạnh mẽ cho Python, hỗ trợ tính toán vector hóa (vectorization) giúp code chạy nhanh hơn nhiều so với vòng lặp thông thường. Ví dụ: thay vì dùng vòng for để tính toán từng phần tử, numpy có thể thực hiện phép toán trên toàn bộ mảng cùng lúc.

-   **`matplotlib.pyplot`**: Thư viện vẽ đồ thị chuyên nghiệp trong Python, cho phép trực quan hóa dữ liệu và kết quả một cách trực quan, dễ hiểu.

-   **Biến `X`**: Mảng numpy chứa **6 giá trị đặc trưng** (features). Đây là biến độc lập trong mô hình, có thể hiểu là các giá trị đầu vào để dự đoán.

-   **Biến `y`**: Mảng numpy chứa **6 nhãn** (labels) tương ứng với từng giá trị trong X. Với bài toán phân loại nhị phân (binary classification), y chỉ nhận 2 giá trị:

    -   `0`: Thuộc lớp âm (negative class)
    -   `1`: Thuộc lớp dương (positive class)

-   **Biến `w` (weight/trọng số)**: Là hệ số góc của đường phân chia trong không gian đặc trưng. Khởi tạo = 0 nghĩa là đường thẳng ban đầu nằm ngang, chưa có khả năng phân loại.

-   **Biến `b` (bias)**: Là hệ số chặn (intercept), xác định vị trí của đường phân chia dịch chuyển lên/xuống. Khởi tạo = 0 nghĩa là đường thẳng đi qua gốc tọa độ.

-   **Biến `alpha` (learning rate/tốc độ học)**: Là bước nhảy khi cập nhật tham số trong thuật toán Gradient Descent. Giá trị 0.0001 khá nhỏ, giúp mô hình học chậm nhưng ổn định, tránh overshooting (nhảy quá xa khỏi điểm tối ưu).

---

### 2️⃣ Hàm Sigmoid - Activation Function

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Giải thích:**

Hàm **sigmoid** (còn gọi là **logistic function**) là trái tim của Logistic Regression. Đây là một hàm kích hoạt (activation function) có vai trò cực kỳ quan trọng:

**Công thức toán học:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Ý nghĩa và đặc điểm:**

1. **Biến đổi giá trị:** Hàm sigmoid nhận đầu vào `z` có thể là bất kỳ số thực nào (từ âm vô cùng đến dương vô cùng) và **nén** (compress) nó vào khoảng **(0, 1)**.

2. **Diễn giải xác suất:** Do giá trị đầu ra luôn nằm trong khoảng (0, 1), sigmoid hoàn hảo để biểu diễn **xác suất**:

    - Đầu ra ≈ 0: Xác suất thuộc lớp 1 rất thấp (gần như chắc chắn thuộc lớp 0)
    - Đầu ra ≈ 0.5: Không chắc chắn, xác suất thuộc lớp 0 và lớp 1 là bằng nhau
    - Đầu ra ≈ 1: Xác suất thuộc lớp 1 rất cao (gần như chắc chắn thuộc lớp 1)

3. **Hình dạng chữ S (S-curve):**

    - Khi z → -∞, σ(z) → 0
    - Khi z = 0, σ(z) = 0.5 (điểm giữa)
    - Khi z → +∞, σ(z) → 1

4. **Tính chất đạo hàm đẹp:** Đạo hàm của sigmoid có dạng σ'(z) = σ(z) × (1 - σ(z)), rất thuận tiện cho việc tính gradient trong quá trình học.

**Ví dụ cụ thể:**

-   sigmoid(0) = 0.5
-   sigmoid(5) ≈ 0.993 (gần 1)
-   sigmoid(-5) ≈ 0.007 (gần 0)

**Trong code:** `np.exp(-z)` tính e^(-z) (e là số Euler ≈ 2.71828). Việc sử dụng numpy giúp tính toán vectorization - có thể truyền vào một mảng z và nhận về một mảng kết quả cùng lúc.

---

### 3️⃣ Hàm Tính Chi Phí (Cost Function)

```python
def compute_cost(X, y, w, b):
    m = len(X) # Số lượng mẫu dữ liệu

    # Bước 1: Tính giá trị tuyến tính z = w*x + b
    z = w * X + b

    # Bước 2: Đưa qua hàm sigmoid để có giá trị dự đoán h (hypothesis)
    h = sigmoid(z)

    # Bước 3: Tính lỗi (Loss) bằng công thức Binary Cross-Entropy
    # Lưu ý: Thêm 1e-15 (epsilon) để tránh lỗi toán học log(0) nếu h=0 hoặc h=1
    cost = -(1/m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
    return cost
```

**Giải thích:**

Đây là hàm cốt lõi để đánh giá **độ tốt** của mô hình Logistic Regression. Hàm này tính toán **Binary Cross-Entropy Loss** - một độ đo chuẩn cho bài toán phân loại nhị phân.

**Chi tiết từng bước:**

#### **Bước 1: Tính giá trị tuyến tính**

```python
z = w * X + b
```

-   **Công thức:** $z = w \cdot x + b$ (tích vô hướng)
-   Đây là phương trình đường thẳng cơ bản trong không gian 1 chiều
-   Với X là mảng, phép toán này được **vector hóa**: mọi phần tử trong X đều được nhân với w và cộng với b
-   **Ví dụ:** Nếu w=2, b=1, X=[1, 2, 3] thì z=[3, 5, 7]

#### **Bước 2: Áp dụng hàm sigmoid**

```python
h = sigmoid(z)
```

-   Biến đổi z thành xác suất dự đoán h (hypothesis)
-   h đại diện cho $P(y=1|x;w,b)$ - xác suất để y=1 khi biết x với tham số w, b
-   Giá trị h nằm trong khoảng (0, 1)

#### **Bước 3: Tính Binary Cross-Entropy**

```python
cost = -(1/m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
```

**Công thức toán học đầy đủ:**

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h^{(i)}) + (1-y^{(i)}) \log(1-h^{(i)}) \right]$$

**Phân tích công thức:**

1. **Trường hợp y = 1 (thực tế là lớp dương):**

    - Phần đóng góp: $-\log(h)$
    - Nếu h gần 1 (dự đoán đúng): $-\log(1) = 0$ → chi phí thấp ✓
    - Nếu h gần 0 (dự đoán sai): $-\log(0) = +\infty$ → chi phí rất cao ✗

2. **Trường hợp y = 0 (thực tế là lớp âm):**

    - Phần đóng góp: $-\log(1-h)$
    - Nếu h gần 0 (dự đoán đúng): $-\log(1-0) = 0$ → chi phí thấp ✓
    - Nếu h gần 1 (dự đoán sai): $-\log(0) = +\infty$ → chi phí rất cao ✗

3. **Epsilon (1e-15):**

    - Là một giá trị cực nhỏ (0.000000000000001) được thêm vào để **tránh lỗi toán học**
    - Khi h = 0 hoặc h = 1, log(0) không xác định (undefined)
    - Thêm epsilon đảm bảo log luôn tính được: log(0 + 1e-15) ≈ -34.5 (số âm lớn nhưng hữu hạn)

4. **Trung bình (1/m):**
    - Chia cho m để lấy trung bình chi phí trên tất cả các mẫu
    - Giúp so sánh công bằng giữa các tập dữ liệu có kích thước khác nhau

**Tại sao dùng Cross-Entropy thay vì Mean Squared Error?**

-   MSE: $(h-y)^2$ → không lồi (non-convex) với sigmoid, nhiều local minimum
-   Cross-Entropy: → hàm lồi (convex), có 1 global minimum duy nhất
-   Gradient Descent với Cross-Entropy hội tụ nhanh và ổn định hơn

---

### 4️⃣ Phần In Kết Quả Và Visualization

#### **4.1. Phần Header và Tính Cost**

```python
print("=" * 60)
print("BÀI 1 - CÂU A: Tính J(w,b)")
print("=" * 60)
print(f"Tham số ban đầu: w = {w}, b = {b}, alpha = {alpha}")
print()
J_wb = compute_cost(X, y, w, b)
```

**Giải thích:**

-   In ra tiêu đề và thông số ban đầu để người đọc dễ theo dõi
-   Gọi hàm `compute_cost(X, y, w, b)` để tính giá trị hàm chi phí với tham số ban đầu
-   Kết quả được lưu vào biến `J_wb` (J of w and b)

#### **4.2. Tạo Figure với 2 Subplots**

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
```

**Giải thích:**

-   Tạo một khung hình (figure) chứa **2 biểu đồ con** (subplots) nằm ngang
-   `1, 2`: 1 hàng, 2 cột
-   `figsize=(14, 5)`: Kích thước 14 inch chiều rộng, 5 inch chiều cao
-   `ax1`: Biểu đồ bên trái (dữ liệu và sigmoid)
-   `ax2`: Biểu đồ bên phải (cost function surface)

---

#### **4.3. Biểu Đồ 1: Dữ Liệu và Sigmoid Function**

```python
# Tạo dải giá trị x mượt mà từ 0 đến 3.5 để vẽ đường cong sigmoid
x_plot = np.linspace(0, 3.5, 100)
z_plot = w * x_plot + b
y_plot = sigmoid(z_plot)
```

**Giải thích:**

-   `np.linspace(0, 3.5, 100)`: Tạo 100 điểm cách đều nhau từ 0 đến 3.5
-   Tính z và y tương ứng để vẽ đường cong sigmoid **mượt mà** (không bị góc cạnh)

```python
# Vẽ các điểm dữ liệu thực tế
ax1.scatter(X[y == 0], y[y == 0], color='blue', s=150, marker='o',
            label='Class 0 (y=0)', edgecolors='black', linewidth=2)
ax1.scatter(X[y == 1], y[y == 1], color='red', s=150, marker='s',
            label='Class 1 (y=1)', edgecolors='black', linewidth=2)
```

**Giải thích:**

-   `X[y == 0]`: Lọc các điểm có nhãn y = 0 (lớp âm)

    -   Vẽ màu **xanh**, hình **tròn** (marker='o')
    -   `s=150`: Kích thước điểm
    -   `edgecolors='black'`: Viền màu đen để dễ phân biệt

-   `X[y == 1]`: Lọc các điểm có nhãn y = 1 (lớp dương)
    -   Vẽ màu **đỏ**, hình **vuông** (marker='s')
    -   Giúp phân biệt rõ ràng 2 lớp

```python
# Vẽ đường dự đoán Sigmoid
ax1.plot(x_plot, y_plot, 'g-', linewidth=2.5,
         label=f'Sigmoid: h(x) = σ({w}x + {b})')
```

**Giải thích:**

-   Vẽ đường cong sigmoid với:
    -   `'g-'`: Màu xanh lá, đường liền
    -   `linewidth=2.5`: Độ dày 2.5
-   **Với w=0, b=0:** Đường sigmoid sẽ là đường thẳng ngang tại y=0.5 (vì z=0 → sigmoid(0)=0.5 cho mọi x)

```python
# Vẽ đường biên quyết định (Decision Boundary)
ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=2,
            label='Decision Boundary (h=0.5)')
```

**Giải thích:**

-   `axhline`: Vẽ đường ngang (horizontal line)
-   **Decision Boundary** tại h=0.5: Ngưỡng phân loại
    -   Nếu h ≥ 0.5 → dự đoán y=1
    -   Nếu h < 0.5 → dự đoán y=0
-   Đường này giúp thấy rõ mô hình đang phân loại các điểm như thế nào

```python
# Gắn nhãn toạ độ lên từng điểm
for i, (xi, yi) in enumerate(zip(X, y)):
    ax1.annotate(f'({xi}, {yi})', (xi, yi),
                textcoords="offset points", xytext=(0,10),
                ha='center', fontsize=9)
```

**Giải thích:**

-   Duyệt qua từng cặp (xi, yi) trong dữ liệu
-   `annotate`: Gắn nhãn văn bản lên biểu đồ
-   `xytext=(0,10)`: Đặt text ở vị trí cách điểm dữ liệu 10 pixels về phía trên
-   `ha='center'`: Căn giữa text theo chiều ngang
-   Giúp người xem dễ đọc giá trị chính xác của từng điểm

```python
# Trang trí biểu đồ 1
ax1.set_xlabel('x', fontsize=13, fontweight='bold')
ax1.set_ylabel('y', fontsize=13, fontweight='bold')
ax1.set_title(f'Dữ liệu và Sigmoid Function\nJ(w={w}, b={b}) = {J_wb:.8f}',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([-0.1, 1.1])
ax1.set_xlim([0, 3.5])
```

**Giải thích:**

-   Đặt nhãn trục x, y với font đậm (bold), kích thước 13
-   Tiêu đề hiển thị giá trị J(w,b) với 8 chữ số thập phân
-   `legend`: Hiển thị chú thích các đường/điểm, tự động tìm vị trí tốt nhất (loc='best')
-   `grid`: Lưới nền với độ trong suốt 0.3, đường gạch ngang
-   `set_ylim/xlim`: Giới hạn trục để biểu đồ thoáng, không bị sát mép

---

#### **4.4. Biểu Đồ 2: Mặt Phẳng Cost Function**

```python
# Tạo lưới toạ độ (mesh grid) cho w và b
w_range = np.linspace(-2, 2, 50)
b_range = np.linspace(-2, 2, 50)
W, B = np.meshgrid(w_range, b_range)
Z = np.zeros_like(W)
```

**Giải thích:**

-   Tạo 50 giá trị w từ -2 đến 2
-   Tạo 50 giá trị b từ -2 đến 2
-   `meshgrid`: Tạo lưới 2D, mỗi điểm trên lưới là một cặp (w, b)
-   `Z`: Ma trận 50×50 để lưu giá trị Cost tại mỗi điểm (w, b)

```python
# Tính Cost cho từng điểm trên lưới
for i in range(len(w_range)):
    for j in range(len(b_range)):
        Z[j, i] = compute_cost(X, y, W[j, i], B[j, i])
```

**Giải thích:**

-   Duyệt qua tất cả 2500 cặp (w, b) trên lưới
-   Tính Cost tại mỗi điểm và lưu vào ma trận Z
-   Tạo ra "bản đồ địa hình" của hàm Cost: vùng trũng là điểm tối ưu

```python
# Vẽ đường đồng mức (Contour plot)
contour = ax2.contour(W, B, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
```

**Giải thích:**

-   `contour`: Vẽ đường đồng mức (như đường bình độ trên bản đồ địa lý)
-   `levels=20`: Vẽ 20 đường mức khác nhau
-   `cmap='viridis'`: Bảng màu từ tím đậm (cao) đến vàng (thấp)
-   `clabel`: Hiển thị số liệu trên đường đồng mức để biết giá trị Cost

```python
# Đánh dấu vị trí hiện tại (w=0, b=0)
ax2.plot(w, b, 'r*', markersize=20, label=f'(w={w}, b={b})')
```

**Giải thích:**

-   Vẽ ngôi sao đỏ tại vị trí (w=0, b=0)
-   Cho thấy điểm khởi đầu đang ở đâu trên "bản đồ" Cost
-   Giúp hiểu vị trí hiện tại so với điểm tối ưu (vùng trũng nhất)

---

#### **4.5. Lưu và Hiển Thị**

```python
plt.tight_layout()
plt.savefig('results/ex1a_cost_function_visualization.png',
            dpi=300, bbox_inches='tight')
plt.show()
```

**Giải thích:**

-   `tight_layout()`: Tự động điều chỉnh khoảng cách giữa các subplot để không bị chồng lấn
-   `savefig`: Lưu hình với độ phân giải cao (300 DPI - chất lượng in ấn)
-   `bbox_inches='tight'`: Cắt bỏ khoảng trắng thừa xung quanh
-   `show()`: Hiển thị biểu đồ lên màn hình

---

#### **4.6. In Chi Tiết Quá Trình Tính Toán**

```python
print(f"Số mẫu dữ liệu (m): {len(X)}")
print(f"\nDữ liệu từng điểm:")

for i, (xi, yi) in enumerate(zip(X, y)):
    z_i = w * xi + b
    h_i = sigmoid(z_i)
    print(f"  x[{i}] = {xi}, y[{i}] = {yi} => z = {z_i:.1f}, h(x) = {h_i:.4f}")

print(f"\nCông thức Cost function: J(w,b) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]")
print(f"Kết quả cuối cùng: J({w}, {b}) = {J_wb:.8f}")
print("=" * 60)
```

**Giải thích:**

-   In số lượng mẫu dữ liệu (m = 6)
-   Duyệt qua từng điểm, tính và in:
    -   `z_i`: Giá trị tuyến tính tại điểm thứ i
    -   `h_i`: Xác suất dự đoán sau khi qua sigmoid
-   Hiển thị công thức Cost function để người đọc hiểu rõ
-   In kết quả cuối cùng với 8 chữ số thập phân

---

## 📊 Output và Kết Quả

### 🖥️ Console Output

```
============================================================
BÀI 1 - CÂU A: Tính J(w,b)
============================================================
Tham số ban đầu: w = 0, b = 0, alpha = 0.0001

Số mẫu dữ liệu (m): 6

Dữ liệu từng điểm:
  x[0] = 0.5, y[0] = 0 => z = 0.0, h(x) = 0.5000
  x[1] = 1.0, y[1] = 0 => z = 0.0, h(x) = 0.5000
  x[2] = 1.5, y[2] = 0 => z = 0.0, h(x) = 0.5000
  x[3] = 3.0, y[3] = 1 => z = 0.0, h(x) = 0.5000
  x[4] = 2.0, y[4] = 1 => z = 0.0, h(x) = 0.5000
  x[5] = 1.0, y[5] = 1 => z = 0.0, h(x) = 0.5000

Công thức Cost function: J(w,b) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
Kết quả cuối cùng: J(0, 0) = 0.69314718
============================================================
```

### 📈 Phân Tích Kết Quả

#### **1. Giá trị J(0, 0) = 0.69314718**

Đây là giá trị hàm chi phí khi mô hình **chưa được huấn luyện** (w=0, b=0):

-   **Ý nghĩa:** Mô hình đang dự đoán xác suất 0.5 cho mọi điểm (hoàn toàn ngẫu nhiên)
-   **So sánh với log(2):**
    -   $\ln(2) = 0.693147...$
    -   Giá trị J(0,0) gần bằng ln(2) không phải ngẫu nhiên!

**Giải thích toán học:**

Khi h = 0.5 cho mọi điểm:

$$J = -\frac{1}{m}\sum_{i=1}^{m}[y_i\log(0.5) + (1-y_i)\log(0.5)]$$

$$= -\frac{1}{m}\sum_{i=1}^{m}\log(0.5)$$

$$= -\log(0.5) = \log(2) \approx 0.693$$

Đây là **chi phí tối đa** của một mô hình phân loại nhị phân khi dự đoán hoàn toàn ngẫu nhiên (50-50).

#### **2. Tại sao h(x) = 0.5 cho mọi x?**

Với w=0, b=0:

-   $z = 0 \cdot x + 0 = 0$ (cho mọi x)
-   $h(x) = \sigma(0) = \frac{1}{1+e^0} = \frac{1}{2} = 0.5$

Đường sigmoid là đường **thẳng ngang** tại y=0.5, không có khả năng phân loại.

#### **3. Biểu đồ 1: Dữ liệu và Sigmoid**

Biểu đồ này cho thấy:

-   **Các điểm xanh (y=0)** ở phía trái (x nhỏ)
-   **Các điểm đỏ (y=1)** ở phía phải (x lớn)
-   **Đường sigmoid màu xanh lá** là đường ngang tại 0.5 (do w=0, b=0)
-   **Đường cam gạch ngang** là decision boundary (h=0.5)

**Nhận xét:** Mô hình hiện tại **không phân loại được gì** vì tất cả điểm đều được dự đoán xác suất 0.5.

#### **4. Biểu đồ 2: Cost Function Surface**

Biểu đồ này thể hiện "địa hình" của hàm Cost trong không gian (w, b):

-   **Màu tím đậm:** Vùng có Cost cao (mô hình tệ)
-   **Màu vàng:** Vùng có Cost thấp (mô hình tốt)
-   **Ngôi sao đỏ tại (0,0):** Vị trí ban đầu
-   **Vùng trũng (valley):** Hướng đến điểm tối ưu

**Quan sát:**

-   Điểm (0,0) nằm ở vùng có Cost ≈ 0.693 (không phải tệ nhất nhưng cũng không tốt)
-   Có một vùng trũng rõ ràng hướng về phía w dương, b âm
-   Đây là hướng mà Gradient Descent sẽ đi để giảm Cost

---
