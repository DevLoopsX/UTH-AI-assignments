import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.5, 1, 1.5, 3, 2, 1])
y = np.array([0, 0, 0, 1, 1, 1])

# Khởi tạo tham số ban đầu cho thuật toán
w = 0
b = 0
alpha = 0.0001

# Hàm sigmoid: Chuyển đổi giá trị z sang xác suất trong khoảng (0, 1)
# Công thức: σ(z) = 1 / (1 + e^(-z))
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hàm tính J(w,b) - Cost function (Hàm chi phí Binary Cross-Entropy)
# Mục tiêu: Đo lường sai số giữa dự đoán (h) và thực tế (y)
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


print("=" * 60)
print("BÀI 1 - CÂU A: Tính J(w,b)")
print("=" * 60)
print(f"Tham số ban đầu: w = {w}, b = {b}, alpha = {alpha}")
print()
J_wb = compute_cost(X, y, w, b)

# Tạo khung hình chứa 2 biểu đồ con (subplots) nằm ngang
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- BIỂU ĐỒ 1: Dữ liệu và hàm Sigmoid hiện tại ---
# Tạo dải giá trị x mượt mà từ 0 đến 3.5 để vẽ đường cong sigmoid
x_plot = np.linspace(0, 3.5, 100)
z_plot = w * x_plot + b
y_plot = sigmoid(z_plot) # Tính giá trị sigmoid tương ứng

# Vẽ các điểm dữ liệu thực tế
# Điểm thuộc lớp 0 (y=0): Màu xanh, hình tròn
ax1.scatter(X[y == 0], y[y == 0], color='blue', s=150, marker='o',
            label='Class 0 (y=0)', edgecolors='black', linewidth=2)
# Điểm thuộc lớp 1 (y=1): Màu đỏ, hình vuông
ax1.scatter(X[y == 1], y[y == 1], color='red', s=150, marker='s',
            label='Class 1 (y=1)', edgecolors='black', linewidth=2)

# Vẽ đường dự đoán Sigmoid (với w=0, b=0 nó sẽ là đường thẳng ngang tại 0.5)
ax1.plot(x_plot, y_plot, 'g-', linewidth=2.5,
         label=f'Sigmoid: h(x) = σ({w}x + {b})')

# Vẽ đường biên quyết định (Decision Boundary) tại ngưỡng 0.5
ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=2,
            label='Decision Boundary (h=0.5)')

# Gắn nhãn toạ độ (x, y) lên từng điểm dữ liệu để dễ nhìn
for i, (xi, yi) in enumerate(zip(X, y)):
    ax1.annotate(f'({xi}, {yi})', (xi, yi),
                textcoords="offset points", xytext=(0,10),
                ha='center', fontsize=9)

# Trang trí biểu đồ 1
ax1.set_xlabel('x', fontsize=13, fontweight='bold')
ax1.set_ylabel('y', fontsize=13, fontweight='bold')
ax1.set_title(f'Dữ liệu và Sigmoid Function\nJ(w={w}, b={b}) = {J_wb:.8f}',
              fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([-0.1, 1.1]) # Giới hạn trục y rộng hơn 0-1 chút để thoáng
ax1.set_xlim([0, 3.5])

# --- BIỂU ĐỒ 2: Mặt phẳng hàm chi phí (Cost Function Surface) ---
# Mục đích: Xem điểm (w=0, b=0) đang đứng ở đâu so với vùng trũng nhất (tối ưu nhất)

# Tạo lưới toạ độ (mesh grid) cho w và b trong khoảng -2 đến 2
w_range = np.linspace(-2, 2, 50)
b_range = np.linspace(-2, 2, 50)
W, B = np.meshgrid(w_range, b_range)
Z = np.zeros_like(W) # Ma trận Z chứa giá trị Cost tại mỗi cặp (w,b)

# Tính Cost cho từng điểm trên lưới
for i in range(len(w_range)):
    for j in range(len(b_range)):
        Z[j, i] = compute_cost(X, y, W[j, i], B[j, i])

# Vẽ đường đồng mức (Contour plot) để thể hiện độ cao của hàm Cost
contour = ax2.contour(W, B, Z, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8) # Hiển thị số liệu trên đường đồng mức

# Đánh dấu vị trí hiện tại (w=0, b=0) bằng ngôi sao màu đỏ
ax2.plot(w, b, 'r*', markersize=20, label=f'(w={w}, b={b})')

# Trang trí biểu đồ 2
ax2.set_xlabel('w', fontsize=13, fontweight='bold')
ax2.set_ylabel('b', fontsize=13, fontweight='bold')
ax2.set_title(f'Cost Function J(w,b)\nJ({w},{b}) = {J_wb:.8f}',
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# Lưu và hiển thị biểu đồ
plt.tight_layout()
# Lưu ý: Bạn cần đảm bảo thư mục đường dẫn tồn tại trước khi chạy lệnh savefig
plt.savefig('results/ex1a_cost_function_visualization.png',
            dpi=300, bbox_inches='tight')
plt.show()

print(f"Số mẫu dữ liệu (m): {len(X)}")
print(f"\nDữ liệu từng điểm:")

# Duyệt qua từng điểm để in chi tiết quá trình tính toán
for i, (xi, yi) in enumerate(zip(X, y)):
    z_i = w * xi + b
    h_i = sigmoid(z_i)
    # z_i: giá trị trước activation, h_i: xác suất dự đoán
    print(f"  x[{i}] = {xi}, y[{i}] = {yi} => z = {z_i:.1f}, h(x) = {h_i:.4f}")

print(f"\nCông thức Cost function: J(w,b) = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]")
print(f"Kết quả cuối cùng: J({w}, {b}) = {J_wb:.8f}")
print("=" * 60)