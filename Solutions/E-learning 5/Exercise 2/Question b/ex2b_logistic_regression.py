import math
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# ========== DỮ LIỆU ==========
dataset = [
    [0.5, 0],  # 0.5 giờ học → Rớt
    [1.0, 0],  # 1.0 giờ học → Rớt
    [1.5, 0],  # 1.5 giờ học → Rớt
    [2.0, 0],  # 2.0 giờ học → Rớt
    [2.5, 1],  # 2.5 giờ học → Đậu
    [3.0, 1],  # 3.0 giờ học → Đậu
    [3.5, 1],  # 3.5 giờ học → Đậu
    [4.0, 1]   # 4.0 giờ học → Đậu
]

# ========== CÁC HÀM CHO MÔ HÌNH TỰ XÂY DỰNG ==========

def get_prediction(m, b, x):
    z = m * x + b
    return 1 / (1 + math.exp(-z))

def get_cost(y, y_hat):
    n = len(y)
    total_cost = 0.0
    for yi, y_hat_i in zip(y, y_hat):
        total_cost += -(yi * math.log(y_hat_i) + (1 - yi) * math.log(1 - y_hat_i))
    return total_cost / n

def get_gradients(m, b, x, y, y_hat):
    n = len(y)
    dm = (1 / n) * sum((y_hat_i - yi) * xi for y_hat_i, yi, xi in zip(y_hat, y, x))
    db = (1 / n) * sum(y_hat_i - yi for y_hat_i, yi in zip(y_hat, y))
    return dm, db

# ========== PHẦN A: MÔ HÌNH TỰ XÂY DỰNG ==========

# Khởi tạo tham số
m = 1.0
b = -1.0

# Siêu tham số
iterations = 10
learning_rate = 1.0

# Tách dữ liệu
x = [row[0] for row in dataset]  # Số giờ học
y = [row[1] for row in dataset]  # Kết quả (0: rớt, 1: đậu)

# Huấn luyện mô hình bằng Gradient Descent
for it in range(iterations):
    # Forward pass: Tính dự đoán
    y_hat = [get_prediction(m, b, xi) for xi in x]

    # Tính cost (không bắt buộc cho huấn luyện, chỉ để theo dõi)
    cost = get_cost(y, y_hat)

    # Backward pass: Tính gradient
    dm, db = get_gradients(m, b, x, y, y_hat)

    # Update parameters
    m -= learning_rate * dm
    b -= learning_rate * db

# Dự đoán cho sinh viên học 2.8 giờ
hours_input = 2.8
predicted_score_manual = get_prediction(m, b, hours_input)

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

# ========== PHẦN B: MÔ HÌNH SKLEARN ==========

print("\n" + "-"*60)
print("PHẦN B: KẾT QUẢ MÔ HÌNH SKLEARN")
print("-"*60)

# Chuẩn bị dữ liệu cho sklearn
X = np.array([[row[0]] for row in dataset])  # Features (Hours)
y_train = np.array([row[1] for row in dataset])  # Labels (Pass)

# Tạo và huấn luyện mô hình Logistic Regression
model = LogisticRegression(max_iter=10, solver='lbfgs', random_state=42)
model.fit(X, y_train)

# Dự đoán với sklearn
X_test = np.array([[hours_input]])
predicted_proba_sklearn = model.predict_proba(X_test)[0][1]  # Xác suất cho class 1 (Pass)
predicted_class_sklearn = model.predict(X_test)[0]

print(f"Tham số học được:")
print(f"  - Hệ số góc (coef): {model.coef_[0][0]:.6f}")
print(f"  - Hệ số chặn (intercept): {model.intercept_[0]:.6f}")
print(f"\nDự đoán cho sinh viên học {hours_input} giờ:")
print(f"  - Xác suất đậu: {predicted_proba_sklearn:.6f} ({predicted_proba_sklearn*100:.2f}%)")
if predicted_class_sklearn == 1:
    print(f"  - Kết luận: ĐẬU")
else:
    print(f"  - Kết luận: RỚT")

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

print("\n" + "="*60)
print("KẾT LUẬN")
print("="*60)
print("Có thể thấy sự khác biệt giữa hai mô hình do:")
print("  1. Số lần lặp khác nhau (10 vs thuật toán tối ưu của sklearn)")
print("  2. Phương pháp tối ưu khác nhau (Gradient Descent vs LBFGS)")
print("  3. Điều kiện dừng và khởi tạo tham số khác nhau")
print("="*60 + "\n")

# ========== VISUALIZATION ==========

# Tạo figure với 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: So sánh decision boundary của 2 mô hình
x_plot = np.linspace(0, 4.5, 100)
y_manual = [get_prediction(m, b, x_i) for x_i in x_plot]
y_sklearn = [model.predict_proba([[x_i]])[0][1] for x_i in x_plot]

# Vẽ dữ liệu gốc
x_data = [row[0] for row in dataset]
y_data = [row[1] for row in dataset]
ax1.scatter(x_data, y_data, c=['red' if y==0 else 'green' for y in y_data],
            s=100, alpha=0.6, edgecolors='black', linewidth=1.5,
            label='Dữ liệu thực tế', zorder=3)

# Vẽ đường sigmoid
ax1.plot(x_plot, y_manual, 'b-', linewidth=2, label='Mô hình tự xây dựng')
ax1.plot(x_plot, y_sklearn, 'r--', linewidth=2, label='Sklearn')

# Vẽ ngưỡng 0.5
ax1.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Ngưỡng 0.5')

# Vẽ điểm dự đoán cho 2.8 giờ
ax1.scatter([hours_input], [predicted_score_manual], c='blue', s=200,
            marker='*', edgecolors='black', linewidth=1.5,
            label=f'Dự đoán {hours_input}h (Manual)', zorder=4)
ax1.scatter([hours_input], [predicted_proba_sklearn], c='red', s=200,
            marker='*', edgecolors='black', linewidth=1.5,
            label=f'Dự đoán {hours_input}h (Sklearn)', zorder=4)

ax1.set_xlabel('Số giờ học', fontsize=11, fontweight='bold')
ax1.set_ylabel('Xác suất đậu', fontsize=11, fontweight='bold')
ax1.set_title('So sánh Decision Boundary\nMô hình tự xây dựng vs Sklearn', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# Subplot 2: So sánh các tham số
categories = ['Hệ số góc\n(m/coef)', 'Hệ số chặn\n(b/intercept)', f'Xác suất đậu\n({hours_input}h)']
manual_values = [m, b, predicted_score_manual]
sklearn_values = [model.coef_[0][0], model.intercept_[0], predicted_proba_sklearn]

x_pos = np.arange(len(categories))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, manual_values, width, label='Mô hình tự xây dựng', color='skyblue', edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x_pos + width/2, sklearn_values, width, label='Sklearn', color='salmon', edgecolor='black', linewidth=1.5)

# Thêm giá trị lên các cột
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel('Giá trị', fontsize=11, fontweight='bold')
ax2.set_title('So sánh Tham số và Kết quả Dự đoán', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(categories, fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('results/ex2b_comparison_chart.png', dpi=300, bbox_inches='tight')
print("📊 Biểu đồ đã được lưu vào 'results/ex2b_comparison_chart.png'")
plt.show()
