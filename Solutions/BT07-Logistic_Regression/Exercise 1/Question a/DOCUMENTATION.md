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

Đoạn code này thực hiện việc chuẩn bị môi trường và khởi tạo dữ liệu cho bài toán Logistic Regression. Đầu tiên, hai thư viện quan trọng được import: `numpy` là thư viện toán học mạnh mẽ hỗ trợ tính toán vector hóa (vectorization), giúp code chạy nhanh hơn nhiều so với vòng lặp thông thường - thay vì dùng vòng for để tính toán từng phần tử, numpy có thể thực hiện phép toán trên toàn bộ mảng cùng lúc; `matplotlib.pyplot` là thư viện vẽ đồ thị chuyên nghiệp trong Python, cho phép trực quan hóa dữ liệu và kết quả một cách trực quan, dễ hiểu.

Tiếp theo, dữ liệu được khởi tạo với biến `X` là mảng numpy chứa 6 giá trị đặc trưng (features) - đây là biến độc lập trong mô hình, có thể hiểu là các giá trị đầu vào để dự đoán. Biến `y` là mảng numpy chứa 6 nhãn (labels) tương ứng với từng giá trị trong X. Với bài toán phân loại nhị phân (binary classification), y chỉ nhận 2 giá trị: `0` thuộc lớp âm (negative class) và `1` thuộc lớp dương (positive class).

Về các tham số của mô hình, biến `w` (weight/trọng số) là hệ số góc của đường phân chia trong không gian đặc trưng - khởi tạo bằng 0 nghĩa là đường thẳng ban đầu nằm ngang, chưa có khả năng phân loại. Biến `b` (bias) là hệ số chặn (intercept), xác định vị trí của đường phân chia dịch chuyển lên/xuống - khởi tạo bằng 0 nghĩa là đường thẳng đi qua gốc tọa độ. Cuối cùng, biến `alpha` (learning rate/tốc độ học) là bước nhảy khi cập nhật tham số trong thuật toán Gradient Descent - giá trị 0.0001 khá nhỏ, giúp mô hình học chậm nhưng ổn định, tránh overshooting (nhảy quá xa khỏi điểm tối ưu).

---

### 2️⃣ Hàm Sigmoid - Activation Function

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Giải thích:**

Hàm sigmoid (còn gọi là logistic function) là trái tim của Logistic Regression, đây là một hàm kích hoạt (activation function) có vai trò cực kỳ quan trọng. Hàm này hoạt động theo công thức toán học:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Trong đó `z` có thể là bất kỳ số thực nào từ âm vô cùng đến dương vô cùng, và hàm sigmoid sẽ "nén" (compress) giá trị này vào khoảng (0, 1).

Đặc điểm quan trọng nhất của sigmoid là khả năng diễn giải xác suất. Do giá trị đầu ra luôn nằm trong khoảng (0, 1), sigmoid hoàn hảo để biểu diễn xác suất: khi đầu ra gần 0 nghĩa là xác suất thuộc lớp 1 rất thấp (gần như chắc chắn thuộc lớp 0), khi đầu ra gần 0.5 thì không chắc chắn (xác suất thuộc lớp 0 và lớp 1 bằng nhau), và khi đầu ra gần 1 thì xác suất thuộc lớp 1 rất cao (gần như chắc chắn thuộc lớp 1).

Hàm sigmoid có hình dạng chữ S (S-curve) đặc trưng. Khi z tiến về âm vô cùng thì σ(z) tiến về 0, khi z = 0 thì σ(z) = 0.5 (điểm giữa), và khi z tiến về dương vô cùng thì σ(z) tiến về 1. Ví dụ cụ thể, sigmoid(0) = 0.5, sigmoid(5) ≈ 0.993 (gần 1), và sigmoid(-5) ≈ 0.007 (gần 0). Một tính chất quan trọng khác là đạo hàm của sigmoid có dạng:

$$\sigma'(z) = \sigma(z) \times (1 - \sigma(z))$$

Điều này rất thuận tiện cho việc tính gradient trong quá trình học.

Trong code, `np.exp(-z)` tính e^(-z) với e là số Euler (≈ 2.71828). Việc sử dụng numpy giúp tính toán vectorization, nghĩa là có thể truyền vào một mảng z và nhận về một mảng kết quả cùng lúc, giúp code chạy nhanh hơn rất nhiều.

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

Đây là hàm cốt lõi để đánh giá độ tốt của mô hình Logistic Regression, hàm này tính toán Binary Cross-Entropy Loss - một độ đo chuẩn cho bài toán phân loại nhị phân. Quá trình tính toán diễn ra qua ba bước chính.

Bước đầu tiên là tính giá trị tuyến tính theo công thức:

$$z = w \cdot x + b$$

Đây là phương trình đường thẳng cơ bản trong không gian 1 chiều. Với X là mảng, phép toán này được vector hóa, nghĩa là mọi phần tử trong X đều được nhân với w và cộng với b cùng lúc. Ví dụ, nếu w=2, b=1, và X=[1, 2, 3] thì kết quả z=[3, 5, 7].

Bước thứ hai là áp dụng hàm sigmoid để biến đổi z thành xác suất dự đoán h (hypothesis). Giá trị h này đại diện cho:

$$P(y=1|x;w,b)$$

Đây là xác suất để y=1 khi biết x với tham số w, b. Giá trị h luôn nằm trong khoảng (0, 1).

Bước cuối cùng là tính Binary Cross-Entropy theo công thức:

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h^{(i)}) + (1-y^{(i)}) \log(1-h^{(i)}) \right]$$

Công thức này hoạt động thông minh ở chỗ: khi y=1 (thực tế là lớp dương), phần đóng góp là:

$$-\log(h)$$

Nếu h gần 1 (dự đoán đúng) thì cho chi phí thấp, nhưng nếu h gần 0 (dự đoán sai) thì cho chi phí rất cao. Tương tự, khi y=0 (thực tế là lớp âm), phần đóng góp là:

$$-\log(1-h)$$

Nếu h gần 0 (dự đoán đúng) thì cho chi phí thấp, còn nếu h gần 1 (dự đoán sai) thì cho chi phí rất cao.

Trong code, epsilon (1e-15) là một giá trị cực nhỏ (0.000000000000001) được thêm vào để tránh lỗi toán học khi tính log(0). Khi h = 0 hoặc h = 1, log(0) không xác định (undefined), nhưng thêm epsilon đảm bảo log luôn tính được: log(0 + 1e-15) ≈ -34.5 (số âm lớn nhưng hữu hạn). Việc chia cho m (số mẫu) để lấy trung bình chi phí trên tất cả các mẫu giúp so sánh công bằng giữa các tập dữ liệu có kích thước khác nhau.

Một câu hỏi thường gặp là tại sao dùng Cross-Entropy thay vì Mean Squared Error (MSE). Lý do là MSE có dạng:

$$(h-y)^2$$

Công thức này tạo ra hàm không lồi (non-convex) khi kết hợp với sigmoid, dẫn đến nhiều local minimum và khó tối ưu. Trong khi đó, Cross-Entropy tạo ra hàm lồi (convex) có 1 global minimum duy nhất, giúp Gradient Descent hội tụ nhanh và ổn định hơn rất nhiều.

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

Phần code này bắt đầu bằng việc in ra tiêu đề và thông số ban đầu để người đọc dễ theo dõi quá trình thực thi. Sau đó, hàm `compute_cost(X, y, w, b)` được gọi để tính giá trị hàm chi phí với tham số ban đầu w=0 và b=0. Kết quả được lưu vào biến `J_wb` (viết tắt của J of w and b) để sử dụng trong các phần hiển thị tiếp theo. Giá trị này sẽ cho biết mô hình đang hoạt động tệ như thế nào trước khi được huấn luyện.

#### **4.2. Tạo Figure với 2 Subplots**

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
```

**Giải thích:**

Một khung hình (figure) chứa 2 biểu đồ con (subplots) nằm ngang cạnh nhau được tạo ra, với cấu hình 1 hàng và 2 cột. Kích thước của figure được đặt là 14 inch chiều rộng và 5 inch chiều cao (`figsize=(14, 5)`) để đảm bảo các biểu đồ hiển thị rõ ràng và có đủ không gian. Biến `ax1` đại diện cho biểu đồ bên trái sẽ hiển thị dữ liệu và đường sigmoid, trong khi `ax2` là biểu đồ bên phải sẽ thể hiện bề mặt cost function trong không gian (w, b).

---

#### **4.3. Biểu Đồ 1: Dữ Liệu và Sigmoid Function**

```python
# Tạo dải giá trị x mượt mà từ 0 đến 3.5 để vẽ đường cong sigmoid
x_plot = np.linspace(0, 3.5, 100)
z_plot = w * x_plot + b
y_plot = sigmoid(z_plot)
```

**Giải thích:**

Để vẽ được đường cong sigmoid mượt mà, cần tạo một dải giá trị x liên tục. Hàm `np.linspace(0, 3.5, 100)` tạo ra 100 điểm cách đều nhau từ 0 đến 3.5, đủ dày đặc để đường cong không bị góc cạnh hay gãy khúc. Sau đó, giá trị z tương ứng với từng điểm x được tính theo công thức:

$$z = w \times x + b$$

Rồi đưa qua hàm sigmoid để có được giá trị y, tạo thành đường cong hoàn chỉnh để vẽ lên biểu đồ.

```python
# Vẽ các điểm dữ liệu thực tế
ax1.scatter(X[y == 0], y[y == 0], color='blue', s=150, marker='o',
            label='Class 0 (y=0)', edgecolors='black', linewidth=2)
ax1.scatter(X[y == 1], y[y == 1], color='red', s=150, marker='s',
            label='Class 1 (y=1)', edgecolors='black', linewidth=2)
```

**Giải thích:**

Trong phần này, các điểm dữ liệu thực tế được vẽ lên biểu đồ để trực quan hóa dữ liệu. Với các điểm thuộc lớp 0 (y=0), cú pháp `X[y == 0]` được sử dụng để lọc ra những điểm có nhãn y bằng 0, sau đó vẽ chúng bằng màu xanh dương (blue) với hình tròn (marker='o'). Kích thước điểm được đặt là 150 (`s=150`) để dễ nhìn thấy, và có viền màu đen (`edgecolors='black'`) với độ dày 2 (`linewidth=2`) để tạo sự phân biệt rõ ràng.

Tương tự, với các điểm thuộc lớp 1 (y=1), `X[y == 1]` được sử dụng để lọc ra những điểm có nhãn y bằng 1, và vẽ chúng bằng màu đỏ (red) với hình vuông (marker='s'). Việc sử dụng hai màu khác nhau (xanh cho lớp 0, đỏ cho lớp 1) và hai hình dạng khác nhau (tròn và vuông) giúp người xem dễ dàng phân biệt hai lớp dữ liệu ngay cả khi xem trên ảnh đen trắng hoặc khi có vấn đề về màu sắc.

```python
# Vẽ đường dự đoán Sigmoid
ax1.plot(x_plot, y_plot, 'g-', linewidth=2.5,
         label=f'Sigmoid: h(x) = σ({w}x + {b})')
```

**Giải thích:**

Sau khi vẽ các điểm dữ liệu, đường cong sigmoid thể hiện mô hình dự đoán hiện tại được vẽ. Đường này được vẽ bằng màu xanh lá cây (green) với kiểu đường liền (`'g-'`) và độ dày 2.5 (`linewidth=2.5`) để nổi bật trên biểu đồ. Nhãn (label) của đường hiển thị công thức toán học:

$$h(x) = \sigma(wx + b)$$

Với giá trị w và b hiện tại. Điều đặc biệt là với w=0 và b=0 như trong trường hợp ban đầu, công thức trở thành:

$$\sigma(0)$$

Cho mọi giá trị x, dẫn đến đường sigmoid sẽ là một đường thẳng ngang tại y=0.5 bởi vì sigmoid(0) luôn bằng 0.5 bất kể x là bao nhiêu.

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

Phần code này duyệt qua từng cặp (xi, yi) trong dữ liệu và gắn nhãn văn bản lên biểu đồ bằng hàm `annotate`. Tham số `xytext=(0,10)` xác định vị trí của text cách điểm dữ liệu 10 pixels về phía trên, còn `ha='center'` căn giữa text theo chiều ngang. Việc hiển thị tọa độ trực tiếp trên biểu đồ giúp người xem dễ dàng đọc giá trị chính xác của từng điểm mà không cần ước lượng từ các trục.

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

Phần trang trí này thiết lập các thuộc tính hiển thị cho biểu đồ. Nhãn trục x và y được đặt với font đậm (bold) và kích thước 13 để dễ đọc. Tiêu đề biểu đồ hiển thị công thức và giá trị J(w,b) với 8 chữ số thập phân, cung cấp thông tin chi tiết về hàm chi phí tại tham số hiện tại. Hàm `legend` hiển thị chú thích cho các đường và điểm, với `loc='best'` cho phép matplotlib tự động tìm vị trí tối ưu không bị che khuất dữ liệu. Lưới nền được bật với độ trong suốt 0.3 và kiểu đường gạch ngang, giúp người xem dễ ước lượng giá trị mà không làm loãng biểu đồ. Cuối cùng, `set_ylim` và `set_xlim` giới hạn phạm vi các trục để biểu đồ thoáng đãng, không bị sát mép khung hình.

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

Để vẽ bề mặt Cost function trong không gian 2 chiều (w, b), trước tiên cần tạo lưới tọa độ (mesh grid). Hàm `np.linspace(-2, 2, 50)` tạo ra 50 giá trị cách đều nhau từ -2 đến 2 cho cả w và b, tạo thành dải giá trị khảo sát. Sau đó, hàm `meshgrid` kết hợp hai dải giá trị này thành lưới 2D, trong đó mỗi điểm trên lưới đại diện cho một cặp tham số (w, b) cụ thể. Ma trận Z được khởi tạo với kích thước 50×50 bằng `zeros_like(W)` để lưu giá trị Cost tại từng điểm (w, b) trên lưới, tạo thành "bản đồ địa hình" của hàm chi phí.

```python
# Tính Cost cho từng điểm trên lưới
for i in range(len(w_range)):
    for j in range(len(b_range)):
        Z[j, i] = compute_cost(X, y, W[j, i], B[j, i])
```

**Giải thích:**

Vòng lặp lồng nhau này duyệt qua tất cả 2500 cặp (w, b) trên lưới (50×50), tính giá trị Cost tại mỗi điểm bằng hàm `compute_cost`, và lưu kết quả vào ma trận Z tại vị trí tương ứng. Quá trình này tạo ra "bản đồ địa hình" hoàn chỉnh của hàm Cost trong không gian tham số, trong đó các vùng trũng (giá trị thấp) đại diện cho các điểm tối ưu, còn các vùng cao (giá trị lớn) đại diện cho các tham số kém hiệu quả.

```python
# Vẽ đường đồng mức (Contour plot)
contour = ax2.contour(W, B, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
```

**Giải thích:**

Hàm `contour` vẽ các đường đồng mức (contour lines) tương tự như đường bình độ trên bản đồ địa lý, trong đó mỗi đường nối các điểm có cùng giá trị Cost. Tham số `levels=20` chỉ định vẽ 20 đường mức khác nhau để thể hiện sự thay đổi của Cost một cách chi tiết. Bảng màu `viridis` được chọn với gradient từ tím đậm (giá trị Cost cao) qua xanh lá sang vàng (giá trị Cost thấp), giúp người xem dễ dàng nhận biết các vùng tối ưu. Hàm `clabel` với `inline=True` hiển thị giá trị số trực tiếp trên các đường đồng mức với cỡ chữ 8, cung cấp thông tin định lượng chính xác về giá trị Cost tại từng vùng.

```python
# Đánh dấu vị trí hiện tại (w=0, b=0)
ax2.plot(w, b, 'r*', markersize=20, label=f'(w={w}, b={b})')
```

**Giải thích:**

Lệnh này vẽ một ngôi sao đỏ kích thước lớn (markersize=20) tại vị trí (w=0, b=0) trên biểu đồ contour, đánh dấu rõ ràng điểm khởi đầu của tham số. Việc hiển thị vị trí hiện tại trên "bản đồ địa hình" Cost giúp người xem dễ dàng so sánh với vùng tối ưu (vùng trũng nhất có Cost thấp), từ đó hiểu được mô hình cần di chuyển theo hướng nào để cải thiện hiệu suất.

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

Phần cuối của chương trình in ra chi tiết quá trình tính toán để người đọc hiểu rõ từng bước. Đầu tiên, số lượng mẫu dữ liệu (m = 6) được in ra để biết quy mô tập dữ liệu. Tiếp theo, vòng lặp được sử dụng để duyệt qua từng cặp (xi, yi) trong dữ liệu:

```python
for i, (xi, yi) in enumerate(zip(X, y)):
```

Hàm `enumerate` giúp có thêm chỉ số i để đánh số các điểm.

Với mỗi điểm dữ liệu, giá trị tuyến tính `z_i = w * xi + b` (giá trị trước khi qua activation function) và `h_i = sigmoid(z_i)` (xác suất dự đoán sau khi qua sigmoid) được tính. Kết quả được in ra với `.1f` cho z (1 chữ số thập phân) và `.4f` cho h (4 chữ số thập phân) giúp người đọc thấy rõ quá trình biến đổi từ dữ liệu đầu vào sang xác suất dự đoán.

Cuối cùng, công thức Cost function được hiển thị dưới dạng ký hiệu toán học:

$$J(w,b) = -\frac{1}{m} \times \Sigma[y \times \log(h) + (1-y) \times \log(1-h)]$$

Để người đọc hiểu rõ phương pháp tính, rồi in kết quả cuối cùng `J(0, 0)` với 8 chữ số thập phân để đảm bảo độ chính xác cao.

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

Đây là giá trị hàm chi phí khi mô hình chưa được huấn luyện với w=0 và b=0. Giá trị này thể hiện mô hình đang dự đoán xác suất 0.5 cho mọi điểm, tức là dự đoán hoàn toàn ngẫu nhiên (50-50). Điều thú vị là J(0,0) gần bằng ln(2) ≈ 0.693147, và đây không phải trùng hợp ngẫu nhiên.

**Giải thích toán học:**

Khi h = 0.5 cho mọi điểm, hàm chi phí được tính như sau:

$$J = -\frac{1}{m}\sum_{i=1}^{m}[y_i\log(0.5) + (1-y_i)\log(0.5)]$$

$$= -\frac{1}{m}\sum_{i=1}^{m}\log(0.5)$$

$$= -\log(0.5) = \log(2) \approx 0.693$$

Kết quả này đại diện cho chi phí tối đa của một mô hình phân loại nhị phân khi dự đoán hoàn toàn ngẫu nhiên (50-50), phản ánh việc mô hình chưa học được bất kỳ thông tin hữu ích nào từ dữ liệu.

#### **2. Tại sao h(x) = 0.5 cho mọi x?**

Với tham số ban đầu w=0 và b=0, giá trị tuyến tính z luôn bằng 0 bất kể giá trị x là bao nhiêu:

$$z = 0 \cdot x + 0 = 0 \text{ (cho mọi x)}$$

Khi đưa qua hàm sigmoid:

$$h(x) = \sigma(0) = \frac{1}{1+e^0} = \frac{1}{2} = 0.5$$

Kết quả này dẫn đến đường sigmoid là một đường thẳng ngang tại y=0.5, hoàn toàn không có khả năng phân loại vì mọi điểm dữ liệu đều nhận được xác suất giống nhau.

#### **3. Biểu đồ 1: Dữ liệu và Sigmoid**

Biểu đồ bên trái thể hiện sự phân bố dữ liệu và đường dự đoán của mô hình. Các điểm xanh (y=0) tập trung ở phía trái với giá trị x nhỏ, trong khi các điểm đỏ (y=1) nằm ở phía phải với giá trị x lớn hơn. Đường sigmoid màu xanh lá hiện tại là một đường ngang tại mức 0.5 do w=0 và b=0, trùng với đường quyết định (decision boundary) màu cam gạch ngang. Nhận xét quan trọng là mô hình hiện tại hoàn toàn không phân loại được gì vì tất cả điểm đều nhận xác suất dự đoán giống nhau là 0.5, không phản ánh sự khác biệt giữa hai lớp dữ liệu.

#### **4. Biểu đồ 2: Cost Function Surface**

Biểu đồ bên phải thể hiện "bản đồ địa hình" của hàm Cost trong không gian (w, b). Các vùng màu tím đậm thể hiện những khu vực có giá trị Cost cao, tương ứng với mô hình hoạt động kém, trong khi các vùng màu vàng biểu thị những vùng có Cost thấp với mô hình tốt hơn. Ngôi sao đỏ tại toạ độ (0,0) đánh dấu vị trí ban đầu của tham số, còn các vùng trũng (valley) chỉ hướng đến điểm tối ưu.

Quan sát trên biểu đồ cho thấy điểm (0,0) nằm ở vùng có Cost xấp xỉ 0.693, không phải tệ nhất nhưng cũng không tốt. Biểu đồ thể hiện rõ ràng một vùng trũng hướng về phía w dương và b âm, đây chính là hướng mà thuật toán Gradient Descent sẽ di chuyển để giảm Cost và tìm điểm tối ưu.

---
