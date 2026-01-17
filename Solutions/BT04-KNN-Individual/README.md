# BT04 KNN Regression

## Mục tiêu bài tập

Bài tập gồm hai yêu cầu. Yêu cầu thứ nhất xây dựng hàm `knn_predictor` để dự đoán Salary tại Experience bằng 6.3. Yêu cầu thứ hai so sánh kết quả dự đoán của hàm tự xây dựng với mô hình `KNeighborsRegressor` trong scikit learn với thiết lập tương đương.

## Dữ liệu sử dụng

Dữ liệu gồm 14 mẫu. Mỗi mẫu có một thuộc tính đầu vào là Experience và một giá trị đầu ra là Salary. Salary trong dữ liệu được biểu diễn theo đơn vị nghìn đô la.

```python
import pandas as pd

data = {
    "Experience": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],
    "Salary":     [0.0, 0.0, 0.0, 0.0, 60.0, 64.0, 55.0, 61.0, 66.0, 83.0, 93.0, 91.0, 98.0, 101.0],
}

df = pd.DataFrame(data)

X_train = df["Experience"].values
y_train = df["Salary"].values
x_query = 6.3
```

Vì Experience là dữ liệu một chiều, khoảng cách giữa hai điểm phụ thuộc trực tiếp vào chênh lệch giữa hai giá trị Experience.

## Câu 1

### Nguyên lý KNN cho hồi quy

KNN cho hồi quy dựa trên giả thiết các điểm có Experience gần nhau thường có Salary gần nhau. Khi cần dự đoán Salary tại Experience bằng 6.3, thuật toán thực hiện lần lượt các bước sau. Bước một tính khoảng cách từ 6.3 đến toàn bộ Experience trong tập huấn luyện. Bước hai chọn k điểm có khoảng cách nhỏ nhất. Bước ba lấy trung bình Salary của k điểm được chọn để làm kết quả dự đoán.

Với dữ liệu một chiều, khoảng cách Euclidean giữa hai điểm `x1` và `x2` được rút gọn thành trị tuyệt đối `|x1 - x2|`.

### Hàm `euclidean_distance`

Hàm `euclidean_distance` định nghĩa khoảng cách giữa hai điểm một chiều. Giá trị trả về là độ lớn của chênh lệch giữa hai giá trị Experience.

```python
def euclidean_distance(x1: float, x2: float) -> float:
    return abs(x1 - x2)
```

Trong KNN, toàn bộ quá trình chọn hàng xóm gần nhất phụ thuộc vào khoảng cách. Nếu phép đo khoảng cách sai, thứ tự hàng xóm bị sai và dự đoán cuối cùng bị sai.

### Hàm `knn_predictor`

Hàm `knn_predictor` nhận vào tập huấn luyện `X_train` và `y_train`, điểm cần dự đoán `x_query`, cùng tham số `k`. Giá trị Salary dự đoán chính là `prediction`. Theo yêu cầu đề bài, giá trị cần trả về là `prediction`. Trong notebook, hàm trả về `prediction` và `neighbors_info`, trong đó `neighbors_info` chứa Experience, Salary và khoảng cách của các hàng xóm gần nhất.

```python
from typing import List, Tuple
import numpy as np

def knn_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_query: float,
    k: int = 3
) -> Tuple[float, List[Tuple[float, float, float]]]:
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(x_query, X_train[i])
        distances.append((i, dist))

    distances.sort(key=lambda x: x[1])

    k_nearest_indices = [idx for idx, _ in distances[:k]]
    k_nearest_salaries = [y_train[i] for i in k_nearest_indices]

    prediction = np.mean(k_nearest_salaries)

    neighbors_info = [
        (X_train[i], y_train[i], distances[j][1])
        for j, (i, _) in enumerate(distances[:k])
    ]

    return prediction, neighbors_info
```

Danh sách `distances` chứa các cặp chỉ số và khoảng cách. Việc lưu chỉ số bảo đảm truy xuất đúng Salary tương ứng trong `y_train`. Sau khi sắp xếp tăng dần theo khoảng cách, các phần tử đầu tiên trong `distances` chính là các hàng xóm gần nhất. Danh sách `k_nearest_indices` lấy đúng k chỉ số đứng đầu. Danh sách `k_nearest_salaries` lấy ra Salary của các chỉ số đó. Quy tắc dự đoán của KNN hồi quy sử dụng trung bình cộng, vì vậy `prediction` được tính bằng trung bình của `k_nearest_salaries`. Phần `neighbors_info` ghi lại nội dung của các hàng xóm được chọn để có thể kiểm tra trực tiếp Experience, Salary và khoảng cách.

### Dự đoán tại Experience bằng 6.3 với k bằng 3

Yêu cầu của câu 1 sử dụng Experience bằng 6.3. Trong notebook, k mặc định là 3 và được dùng để lấy kết quả chính thức.

```python
final_prediction, final_neighbors = knn_predictor(X_train, y_train, x_query, k=3)
```

Với dữ liệu bài tập, ba Experience gần nhất của 6.3 là 6.5, 6.0 và 7.0. Salary tương ứng là 91.0, 93.0 và 98.0. Dự đoán KNN hồi quy bằng trung bình cộng nên Salary dự đoán bằng (91.0 + 93.0 + 98.0) chia 3, tương ứng 94.0 nghìn đô la.

## Câu 2

### Chuẩn bị dữ liệu cho scikit learn

`KNeighborsRegressor` yêu cầu ma trận đầu vào có hai chiều, gồm số mẫu và số thuộc tính. Experience chỉ có một thuộc tính nên dữ liệu cần chuyển về dạng cột.

```python
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

X_train_sklearn = X_train.reshape(-1, 1)
x_query_sklearn = np.array([[x_query]])
```

### Huấn luyện và dự đoán bằng `KNeighborsRegressor`

Mô hình KNN của scikit learn được cấu hình số hàng xóm bằng đúng k của hàm tự xây dựng. KNN không học tham số dạng trọng số. Quá trình `fit` lưu trữ dữ liệu huấn luyện để phục vụ tính khoảng cách và chọn hàng xóm khi dự đoán.

```python
knn_sklearn = KNeighborsRegressor(n_neighbors=3)
knn_sklearn.fit(X_train_sklearn, y_train)
sklearn_pred = knn_sklearn.predict(x_query_sklearn)[0]
```

Khi triển khai đúng, `sklearn_pred` trùng với `final_prediction` cho cùng k. Trong notebook, việc so sánh được thực hiện cho nhiều giá trị k nhằm kiểm tra tính nhất quán của triển khai, đồng thời tính sai khác tuyệt đối giữa hai kết quả để xác nhận trùng khớp về mặt số học.

### Truy xuất hàng xóm gần nhất theo scikit learn

Hàm `kneighbors` trả về khoảng cách và chỉ số của các điểm được chọn làm hàng xóm gần nhất. Dữ liệu này có thể đối chiếu với `neighbors_info` của hàm tự xây dựng để xác nhận hai yếu tố. Yếu tố thứ nhất là danh sách điểm được chọn. Yếu tố thứ hai là giá trị khoảng cách tương ứng.

```python
distances_sklearn, indices_sklearn = knn_sklearn.kneighbors(x_query_sklearn)
```

## Kết luận

Hàm `euclidean_distance` xác định phép đo khoảng cách một chiều. Hàm `knn_predictor` thực hiện đầy đủ các bước tính khoảng cách, sắp xếp, chọn k hàng xóm và lấy trung bình Salary để tạo ra `prediction`. Kết quả dự đoán tại Experience bằng 6.3 với k bằng 3 cho giá trị 94.0 nghìn đô la. Kết quả này được so sánh với `KNeighborsRegressor` trong scikit learn trên cùng cấu hình k để xác nhận tính đúng đắn của triển khai.
