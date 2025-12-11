import sys
import os

# Thêm thư mục cha vào path để import module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logistic_regression_utils import (
    DATASET, get_prediction, train_logistic_regression
)

# Huấn luyện mô hình với n = 10 iterations
m, b, costs = train_logistic_regression(
    dataset=DATASET,
    m_init=1.0,
    b_init=-1.0,
    iterations=10,
    learning_rate=1.0
)

hours_input = 2.8
predicted_score = get_prediction(m, b, hours_input)

print("\n" + "-"*40)

print(f"Kết quả dự đoán cho {hours_input} giờ học:")
print(f"Điểm số dự đoán: {predicted_score:.4f}")
print(f"Xác suất đậu: {predicted_score:.4f} ({predicted_score*100:.2f}%)")

if predicted_score >= 0.5:
    print("=> Kết luận: ĐẬU")
else:
    print("=> Kết luận: RỚT")

print("-"*40 + "\n")