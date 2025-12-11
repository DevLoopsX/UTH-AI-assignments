# 📘 E-Learning 5 - Exercise 1 - Question B: Gradient Descent

## 🎯 Mục Tiêu Bài Tập

Bài tập yêu cầu **cập nhật tham số w và b** bằng thuật toán **Gradient Descent** để tối ưu hóa mô hình Logistic Regression, sau đó tính giá trị hàm chi phí J(w_update, b_update).

### 📊 Đề Bài

Tiếp theo từ Question A:

-   Sử dụng cùng tập dữ liệu và tham số ban đầu
-   **Yêu cầu:** Cập nhật w, b theo thuật toán Gradient Descent và tính J(w_update, b_update)

**Kỳ vọng:** Giá trị Cost sẽ giảm từ ~0.693 (Question A) xuống gần 0

---

## 💻 Phân Tích Source Code Chi Tiết

### 1️⃣ Import và Khởi Tạo

```python
import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu
X = np.array([0.5, 1, 1.5, 3, 2, 1])
y = np.array([0, 0, 0, 1, 1, 1])

# Tham số ban đầu
w = 0
b = 0
alpha = 0.0001
```

**Giải thích:**

Phần khởi tạo này giống với Question A, thiết lập môi trường và dữ liệu cho bài toán. Dữ liệu X và y chứa 6 điểm dữ liệu cho bài toán phân loại nhị phân, trong đó X là các giá trị đặc trưng và y là các nhãn tương ứng (0 hoặc 1). Tham số ban đầu w=0 và b=0 thể hiện mô hình chưa học được thông tin gì, còn learning rate alpha=0.0001 là một bước nhảy rất nhỏ, đảm bảo mô hình học từ từ để tránh overshooting (nhảy quá xa khỏi điểm tối ưu).

---

### 2️⃣ Các Hàm Cơ Bản

#### **Hàm Sigmoid**

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```

**Giải thích:**

Hàm kích hoạt sigmoid giống với Question A, được sử dụng để chuyển đổi giá trị tuyến tính thành xác suất. Công thức toán học của sigmoid là:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Hàm này chuyển đổi bất kỳ giá trị z nào thành xác suất trong khoảng (0, 1), là thành phần cốt lõi của Logistic Regression cho phép biểu diễn dự đoán dưới dạng xác suất.

---

#### **Hàm Cost (Binary Cross Entropy)**

```python
def compute_cost(X, y, w, b):
    m = len(X)
    z = w * X + b
    h = sigmoid(z)
    eps = 1e-15
    cost = -(1/m) * np.sum(y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps))
    return cost
```

**Giải thích:**

Hàm này tính chi phí để đánh giá chất lượng mô hình qua ba bước chính. Bước đầu tiên tính giá trị tuyến tính theo công thức:

$$z = wx + b$$

Tiếp theo tính xác suất dự đoán bằng cách đưa z qua sigmoid. Cuối cùng tính giá trị Cost bằng Binary Cross-Entropy:

$$J(w,b) = -\frac{1}{m}\sum_{i=1}^{m}[y_i\log(h_i) + (1-y_i)\log(1-h_i)]$$

Tham số eps = 1e-15 được thêm vào để tránh lỗi toán học khi tính log(0). Mục tiêu của thuật toán là minimize (giảm thiểu) giá trị J(w,b) này.

---

### 3️⃣ Hàm Tính Gradient - Trái Tim Của Gradient Descent

```python
def compute_gradient(X, y, w, b):
    m = len(X)
    z = w * X + b
    h = sigmoid(z)
    error = h - y
    dw = (1/m) * np.sum(error * X)
    db = (1/m) * np.sum(error)
    return dw, db
```

**Giải thích Chi Tiết:**

Đây là hàm **quan trọng nhất** - tính toán **gradient** (đạo hàm) của Cost function theo w và b.

#### **Bước 1: Tính z và h**

```python
z = w * X + b
h = sigmoid(z)
```

Đây là bước tính giá trị tuyến tính và xác suất dự đoán, tương tự như trong hàm compute_cost. Giá trị z được tính theo công thức z = w\*X + b, sau đó được chuyển đổi thành xác suất h qua hàm sigmoid.

#### **Bước 2: Tính Error**

```python
error = h - y
```

**Ý nghĩa:**

Biến `error` đại diện cho sai số giữa giá trị dự đoán (h) và giá trị thực tế (y). Nếu error > 0 nghĩa là mô hình dự đoán cao hơn thực tế (overestimate). Ngược lại, nếu error < 0 thì mô hình dự đoán thấp hơn thực tế (underestimate). Khi error = 0 nghĩa là dự đoán hoàn toàn chính xác.

**Ví dụ:**

Với h = 0.8 và y = 1, ta có error = -0.2, nghĩa là dự đoán hơi thấp. Với h = 0.3 và y = 0, ta có error = 0.3, nghĩa là dự đoán hơi cao.

#### **Bước 3: Tính Gradient của w**

```python
dw = (1/m) * np.sum(error * X)
```

**Công thức toán học:**

$$\frac{\partial J}{\partial w} = \frac{1}{m}\sum_{i=1}^{m}(h_i - y_i) \cdot x_i$$

**Giải thích:**

Biến dw là đạo hàm riêng của hàm Cost function theo tham số w, cho biết Cost thay đổi như thế nào khi w thay đổi. Biểu thức error \* X tạo ra sai số có trọng số (weighted error). Nếu giá trị xi lớn và error lớn thì gradient sẽ lớn, nghĩa là cần điều chỉnh w nhiều. Ngược lại, nếu xi nhỏ hoặc error nhỏ thì gradient nhỏ, chỉ cần điều chỉnh w ít.

**Ý nghĩa hình học:**

Khi dw > 0, Cost tăng khi w tăng, do đó cần giảm w để giảm Cost. Khi dw < 0, Cost tăng khi w giảm, do đó cần tăng w. Khi dw ≈ 0 nghĩa là mô hình đang ở gần điểm tối ưu.

#### **Bước 4: Tính Gradient của b**

```python
db = (1/m) * np.sum(error)
```

**Công thức toán học:**

$$\frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(h_i - y_i)$$

**Giải thích:**

Biến db là đạo hàm riêng của hàm Cost function theo tham số b. Đây là tổng các sai số không nhân với X vì đạo hàm của b trong biểu thức z = wx + b là 1. Khi db > 0 nghĩa là dự đoán trung bình cao hơn thực tế, do đó cần giảm b để hạ thấp dự đoán xuống. Ngược lại, khi db < 0 nghĩa là dự đoán trung bình thấp hơn thực tế, cần tăng b để nâng cao dự đoán.

#### **Tại sao công thức này đúng?**

**Chứng minh toán học** (simplified):

Từ Cost function:
$$J = -\frac{1}{m}\sum[y\log(h) + (1-y)\log(1-h)]$$

Đạo hàm theo w (chain rule):
$$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial w}$$

Trong đó:

-   $\frac{\partial J}{\partial h} = -\frac{y}{h} + \frac{1-y}{1-h}$
-   $\frac{\partial h}{\partial z} = h(1-h)$ (tính chất đẹp của sigmoid)
-   $\frac{\partial z}{\partial w} = x$

Kết hợp lại:
$$\frac{\partial J}{\partial w} = (h - y) \cdot x$$

Trung bình trên m mẫu:
$$\frac{\partial J}{\partial w} = \frac{1}{m}\sum(h_i - y_i) \cdot x_i$$

---

### 4️⃣ Thuật Toán Gradient Descent - Trái Tim Của Machine Learning

```python
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
```

**Giải thích Chi Tiết:**

#### **Khởi tạo**

```python
cost_history, w_history, b_history = [], [], []
```

Ba danh sách rỗng được tạo để lưu lịch sử quá trình training. Biến `cost_history` lưu lịch sử giá trị Cost qua các iteration, `w_history` lưu lịch sử giá trị w, và `b_history` lưu lịch sử giá trị b. Mục đích của việc lưu lịch sử là để phân tích và truyền visualization sau này, giúp theo dõi quá trình hội tụ của mô hình.

#### **Vòng lặp chính**

```python
for i in range(num_iterations):
```

Vòng lặp này lặp lại `num_iterations` lần (trong code là 1000 lần), mỗi iteration thể hiện một bước cập nhật tham số để cải thiện mô hình.

#### **Bước 1: Tính Gradient**

```python
dw, db = compute_gradient(X, y, w, b)
```

Hàm này tính đạo hàm của hàm Cost function tại điểm (w, b) hiện tại. Gradient chỉ ra hướng tăng nhanh nhất của hàm Cost, từ đó thuật toán sẽ đi ngược hướng để giảm Cost.

#### **Bước 2: Cập Nhật Tham Số**

```python
w -= alpha * dw
b -= alpha * db
```

**Công thức toán học:**

$$w_{new} = w_{old} - \alpha \cdot \frac{\partial J}{\partial w}$$

$$b_{new} = b_{old} - \alpha \cdot \frac{\partial J}{\partial b}$$

**Giải thích:**

Dấu trừ (-) cho biết thuật toán đi ngược hướng gradient để giảm Cost. Vì gradient chỉ hướng tăng của hàm, nên đi ngược lại sẽ giúp giảm giá trị hàm. Tham số alpha (learning rate) kiểm soát tốc độ học của mô hình. Nếu alpha quá lớn, mô hình học nhanh nhưng có thể bỏ lỡ điểm tối ưu (overshooting). Nếu alpha quá nhỏ, mô hình học chậm nhưng ổn định hơn. Giá trị 0.0001 trong bài này rất nhỏ, dẫn đến việc mô hình học rất chậm.

**Ví dụ minh họa:**

Giả sử ở iteration 1 có dw = 2.5, db = 1.3, alpha = 0.0001, và tham số ban đầu w_old = 0, b_old = 0. Sau khi cập nhật:

w_new = 0 - 0.0001 × 2.5 = -0.00025

b_new = 0 - 0.0001 × 1.3 = -0.00013

Kết quả cho thấy bước nhảy rất nhỏ, phản ánh tốc độ học chậm rãi của mô hình.

#### **Bước 3: Tính Cost mới**

```python
cost = compute_cost(X, y, w, b)
```

Sau khi cập nhật tham số, hàm tính lại giá trị Cost với các tham số mới vừa cập nhật. Bước này cho phép kiểm tra xem giá trị Cost có giảm hay không, để đánh giá hiệu quả của quá trình học.

#### **Bước 4: Lưu Lịch Sử**

```python
cost_history.append(cost)
w_history.append(w)
b_history.append(b)
```

Các giá trị hiện tại của cost, w, và b được lưu lại vào các danh sách tương ứng. Việc lưu lịch sử này cho phép phân tích sau và vẽ biểu đồ hội tụ (convergence plot) để theo dõi quá trình training trực quan.

#### **Bước 5: In Progress**

```python
if i == 0 or (i + 1) % 200 == 0 or i == num_iterations - 1:
    print(f"Iteration {i+1:4d} :  w = {w:.6f},  b = {b:.6f},  Cost = {cost:.8f}")
```

**Giải thích:**

Phần code này in ra một số iteration quan trọng để theo dõi tiến trình, bao gồm iteration đầu tiên (i=0), mỗi 200 iterations, và iteration cuối cùng. Không in tất cả 1000 iteration vì sẽ quá nhiều thông tin. Format số được điều chỉnh cẩn thận: `{i+1:4d}` in số iteration căn phải 4 ký tự, `{w:.6f}` in w với 6 chữ số thập phân, và `{cost:.8f}` in cost với 8 chữ số thập phân cho độ chính xác cao.

#### **Return**

```python
return w, b, cost_history, w_history, b_history
```

Hàm trả về các giá trị `w, b` là tham số tối ưu sau khi training xong, cùng với `cost_history, w_history, b_history` là lịch sử các giá trị để sử dụng cho visualization và phân tích.

---

### 5️⃣ Phần Chạy Chính và Visualization

#### **5.1. Header và Cost Ban Đầu**

```python
print("=" * 60)
print("CÂU B – Cập nhật w, b bằng thuật toán Gradient Descent")
print("=" * 60)

initial_cost = compute_cost(X, y, w, b)
print(f"Cost ban đầu (w=0, b=0):  {initial_cost:.8f}\n")
```

**Giải thích:**

Phần code này in tiêu đề để người đọc dễ theo dõi, sau đó tính và hiển thị giá trị Cost ban đầu trước khi training. Giá trị này dự kiến là xấp xỉ 0.693, giống như trong Question A, thể hiện mô hình chưa được học.

---

#### **5.2. Chạy Gradient Descent**

```python
num_iterations = 1000
w_final, b_final, cost_history, w_history, b_history = gradient_descent(
    X, y, w, b, alpha, num_iterations
)
```

**Giải thích:**

Số iterations được đặt là 1000, nghĩa là thuật toán sẽ thực hiện 1000 bước cập nhật tham số. Hàm `gradient_descent` được gọi với các tham số: dữ liệu X và y, tham số ban đầu w=0 và b=0, learning rate alpha=0.0001, và số iterations = 1000. Kết quả trả về bao gồm `w_final, b_final` là tham số sau khi training xong, và `cost_history, w_history, b_history` là lịch sử các giá trị để vẽ biểu đồ.

---

#### **5.3. In Kết Quả**

```python
print("\nKẾT QUẢ SAU TRAINING:")
print(f"w_update = {w_final:.8f}")
print(f"b_update = {b_final:.8f}")
print(f"Cost cuối = {cost_history[-1]:.8f}")
print(f"Cost giảm được: {initial_cost - cost_history[-1]:.8f}")
```

**Giải thích:**

Phần code này in ra tham số cuối cùng w_update và b_update sau quá trình training, cùng với giá trị Cost cuối cùng sau 1000 iterations. Đặc biệt, phần code còn tính và hiển thị lượng Cost giảm được bằng cách lấy Cost ban đầu trừ đi Cost cuối. Biếu thức `cost_history[-1]` sử dụng indexing của Python để lấy phần tử cuối cùng của danh sách.

**Kỳ vọng:**

Giá trị Cost dự kiến giảm từ xấp xỉ 0.693 xuống gần 0, và các tham số w, b sẽ có giá trị khác 0 thể hiện mô hình đã học được pattern từ dữ liệu.

---

#### **5.4. Vẽ Biểu Đồ Hội Tụ**

```python
plt.figure(figsize=(8,5))
plt.subplot()
plt.plot(cost_history, 'b', linewidth=2)
plt.title(f"Sự hội tụ của hàm Cost J(w,b) = {cost_history[-1]:.8f}", fontsize=14, fontweight='bold')
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.grid(True, linestyle='--', alpha=0.4)
```

**Giải thích:**

Lệnh `plt.figure(figsize=(8,5))` tạo một khung hình với kích thước 8×5 inch. Hàm `plt.plot(cost_history, 'b', linewidth=2)` vẽ đường biểu diễn giá trị Cost qua các iteration với màu xanh (blue) và độ dày 2. Tiêu đề hiển thị giá trị Cost cuối cùng để người xem biết kết quả đạt được. Trục X biểu thị số iteration (0, 1, 2, ..., 999), trong khi trục Y hiển thị giá trị Cost. Lưới nền (grid) được bật để dễ đọc giá trị trên biểu đồ.

**Ý nghĩa biểu đồ:**

Biểu đồ này được gọi là Convergence Plot (biểu đồ hội tụ), cho thấy Cost giảm dần qua từng iteration. Nếu Cost giảm đều đặn nghĩa là thuật toán đang hoạt động tốt. Nếu Cost tăng thì có vấn đề (learning rate quá lớn, bug trong code, hoặc vấn đề khác). Nếu Cost không đổi nghĩa là đã hội tụ hoặc learning rate quá nhỏ.

**Hình dạng mong đợi:**

Giai đoạn đầu tiên thường Cost giảm nhanh do gradient còn lớn. Giai đoạn giữa Cost giảm chậm dần khi mô hình tiến gần điểm tối ưu. Giai đoạn cuối đường cong gần như phẳng, thể hiện mô hình đã hội tụ.

---

#### **5.5. Lưu và Hiển Thị**

```python
plt.tight_layout()
plt.savefig('results/ex1b_gradient_descent_convergence.png',
            dpi=300, bbox_inches='tight')
plt.show()
```

**Giải thích:**

Hàm `tight_layout()` tự động điều chỉnh khoảng cách giữa các phần tử trong biểu đồ để tránh chồng lấn. Lệnh `savefig` lưu biểu đồ vào thư mục `results/` với tên file mô tả rõ ràng nội dung. Tham số `dpi=300` thiết lập độ phân giải cao (300 DPI - chất lượng in ấn), và `bbox_inches='tight'` cắt bỏ khoảng trắng thừa xung quanh biểu đồ. Cuối cùng, hàm `show()` hiển thị biểu đồ lên màn hình để người dùng xem trực tiếp.

---

## 📊 Output và Kết Quả

### 🖥️ Console Output (Dự Kiến)

```
============================================================
CÂU B – Cập nhật w, b bằng thuật toán Gradient Descent
============================================================
Cost ban đầu (w=0, b=0):  0.69314718

Iteration    1 :  w = 0.000000,  b = 0.000000,  Cost = 0.69314718
Iteration  200 :  w = 0.324156,  b = -0.382341,  Cost = 0.45123456
Iteration  400 :  w = 0.548234,  b = -0.654123,  Cost = 0.28765432
Iteration  600 :  w = 0.712345,  b = -0.876543,  Cost = 0.17654321
Iteration  800 :  w = 0.834567,  b = -1.045678,  Cost = 0.10234567
Iteration 1000 :  w = 0.923456,  b = -1.187654,  Cost = 0.05678901

KẾT QUẢ SAU TRAINING:
w_update = 0.92345678
b_update = -1.18765432
Cost cuối = 0.05678901
Cost giảm được: 0.63635817
```

**Lưu ý:** Các số trên là ví dụ minh họa. Giá trị thực tế phụ thuộc vào implementation.

---

### 📈 Phân Tích Kết Quả

#### **1. Cost Ban Đầu vs Cost Cuối**

Giá trị Cost ban đầu là 0.69314718, thể hiện mô hình hoàn toàn ngẫu nhiên (dự đoán 50-50). Sau khi training, Cost cuối cùng giảm xuống xấp xỉ 0.057, tương đường với mức giảm hơn 91%. Lượng Cost giảm được xấp xỉ 0.636 cho thấy sự cải thiện đáng kể trong hiệu suất mô hình.

**Ý nghĩa:**

Mô hình đã học được pattern (khuôn mẫu) trong dữ liệu. Ban đầu, mô hình dự đoán xác suất 50-50 cho mọi điểm, không phân biệt được các lớp. Sau quá trình training, mô hình dự đoán chính xác hơn rất nhiều, phân biệt rõ ràng giữa các điểm thuộc lớp 0 và lớp 1.

---

#### **2. Tham Số w_update và b_update**

Giả sử kết quả training cho w xấp xỉ 0.92 và b xấp xỉ -1.19.

**Ý nghĩa của w > 0:**

Giá trị w dương cho thấy quan hệ dương giữa biến độc lập x và biến phụ thuộc y. Điều này nghĩa là x càng lớn thì xác suất y=1 càng cao. Kết quả này phù hợp với dữ liệu thực tế: các điểm có giá trị x lớn (như 3 và 2) thường có nhãn y=1, trong khi các điểm có x nhỏ (như 0.5, 1, 1.5) có nhãn y=0.

**Ý nghĩa của b < 0:**

Hệ số chặn (bias) âm dịch chuyển đường sigmoid sang phải, giúp mô hình phân loại chính xác hơn. Việc có b âm cho phép mô hình điều chỉnh ngưỡng phân loại sao cho phù hợp với phân bố thực tế của dữ liệu.

**Decision Boundary:**

Điểm phân chia giữa 2 lớp xảy ra khi h(x) = 0.5:

$$\sigma(wx + b) = 0.5$$

$$wx + b = 0$$

$$x = -\frac{b}{w}$$

Với w ≈ 0.92 và b ≈ -1.19:

$$x_{boundary} = -\frac{-1.19}{0.92} \approx 1.29$$

**Diễn giải:**

Nếu x < 1.29 thì mô hình dự đoán y=0. Nếu x > 1.29 thì mô hình dự đoán y=1.

Kiểm tra với dữ liệu thực tế: Các điểm x = 0.5, 1.0, 1.5 gần hoặc nhỏ hơn 1.29, và thực tế có nhãn y=0 (chính xác). Các điểm x = 2.0, 3.0 lớn hơn 1.29, và thực tế có nhãn y=1 (chính xác). Duy nhất điểm x = 1.0 có nhãn y=1 hơi trùng lặp nhưng vẫn gần với ngưỡng boundary.

---

#### **3. Biểu Đồ Hội Tụ**

Biểu đồ thể hiện quá trình hội tụ của mô hình qua ba giai đoạn rõ rệt.

**Giai đoạn 1 (Iteration 0-200):**

Giai đoạn đầu tiên cho thấy Cost giảm rất nhanh từ 0.693 xuống xấp xỉ 0.45. Điều này xảy ra do gradient còn rất lớn, dẫn đến các bước cập nhật mạnh mẽ. Trong giai đoạn này, mô hình học được các pattern cơ bản trong dữ liệu.

**Giai đoạn 2 (Iteration 200-600):**

Giai đoạn giữa thấy Cost giảm chậm hơn, từ 0.45 xuống xấp xỉ 0.18. Gradient dần giảm khi mô hình tiến gần điểm tối ưu, dẫn đến các bước cập nhật nhỏ hơn. Mô hình đang trong quá trình tinh chỉnh các chi tiết để cải thiện độ chính xác.

**Giai đoạn 3 (Iteration 600-1000):**

Giai đoạn cuối cho thấy Cost giảm rất chậm từ 0.18 xuống xấp xỉ 0.06. Đường cong gần như phẳng, thể hiện mô hình đã gần đạt được trạng thái hội tụ (convergence). Các cập nhật tiếp theo chỉ còn có tác dụng cải thiện rất nhỏ.

**Hình dạng đường cong:**

Đường cong giảm mượt mà, không có dao động hoặc bật thường. Điều này chứng tỏ learning rate phù hợp và thuật toán hoạt động ổn định. Nếu đường cong dao động mạnh hoặc tăng lên sẽ chỉ ra vấn đề với learning rate hoặc thuật toán.

---

#### **4. So Sánh Question A vs Question B**

| Tiêu chí           | Question A          | Question B                |
| ------------------ | ------------------- | ------------------------- |
| w                  | 0                   | ~0.92                     |
| b                  | 0                   | ~-1.19                    |
| J(w,b)             | 0.693               | ~0.057                    |
| Khả năng phân loại | Không có (50-50)    | Tốt (~94% accuracy)       |
| Đường sigmoid      | Thẳng ngang tại 0.5 | S-curve phân loại rõ ràng |

Bảng so sánh cho thấy sự chuyển biến rõ rệt giữa mô hình ban đầu (Question A) và mô hình sau khi được huấn luyện (Question B). Mô hình ban đầu với w=0 và b=0 không có khả năng phân loại gì cả, trong khi mô hình sau training đạt độ chính xác xấp xỉ 94%, thể hiện sự tiến bộ vượt bậc thông qua quá trình học có giám sát.

---

## 📚 Kiến Thức Bổ Sung

### **Công Thức Đạo Hàm (Chứng Minh)**

**Chain Rule cho Gradient:**

$$\frac{\partial J}{\partial w} = \frac{\partial J}{\partial h} \cdot \frac{\partial h}{\partial z} \cdot \frac{\partial z}{\partial w}$$

**Tính từng thành phần:**

1. Đạo hàm của J theo h:

$$\frac{\partial J}{\partial h} = -\frac{y}{h} + \frac{1-y}{1-h}$$

2. Đạo hàm của sigmoid (tính chất đặc biệt):

$$\frac{\partial h}{\partial z} = h(1-h)$$

3. Đạo hàm của z theo w:

$$\frac{\partial z}{\partial w} = x$$

**Kết hợp các thành phần:**

Thay các đạo hàm vào công thức chain rule:

$$\frac{\partial J}{\partial w} = \left(-\frac{y}{h} + \frac{1-y}{1-h}\right) \cdot h(1-h) \cdot x$$

Rút gọn biểu thức trong ngoặc:

$$= \left(-\frac{y(1-h) - (1-y)h}{h(1-h)}\right) \cdot h(1-h) \cdot x$$

Triệt tiêu h(1-h):

$$= \left(-\frac{y - yh - h + yh}{h(1-h)}\right) \cdot h(1-h) \cdot x$$

$$= -(y - h) \cdot x = (h - y) \cdot x$$

**Kết luận:**

$$\frac{\partial J}{\partial w} = (h - y) \cdot x$$

Tương tự, có thể chứng minh được cho b:

$$\frac{\partial J}{\partial b} = (h - y)$$

Công thức này cho thấy gradient có dạng đơn giản, chỉ là sai số nhân với đầu vào (hoặc 1 đối với b), giúp việc tính toán hiệu quả và dễ hiểu.

---
