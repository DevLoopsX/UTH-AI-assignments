# BT05 K means Clustering
## Mục tiêu bài tập
Bài tập gồm hai yêu cầu. Phần A áp dụng thuật toán K means để phân cụm dữ liệu giao thông với K bằng 3. Phần B sử dụng phương pháp Elbow để xác định K tối ưu với K chạy từ 1 đến 9.
## Dữ liệu sử dụng
Dữ liệu gồm 8 khu vực. Hai thuộc tính dùng để phân cụm là lưu lượng giao thông và diện tích.
```python
import pandas as pd
data = {
    "Ma_KV": ["KV0", "KV1", "KV2", "KV3", "KV4", "KV5", "KV6", "KV7"],
    "Luu_luong_giao_thong": [8000, 3000, 12000, 2000, 5000, 6000, 15000, 4000],
    "Dien_tich_km2": [5, 3, 7, 2, 5.5, 6, 8, 3],
}
df = pd.DataFrame(data)
X_original = df[["Luu_luong_giao_thong", "Dien_tich_km2"]].values
```
`Ma_KV` là mã khu vực dùng cho mục đích hiển thị. Phân cụm được thực hiện trên hai thuộc tính số trong `X_original`.
## Chuẩn hóa dữ liệu
K means sử dụng khoảng cách Euclidean. Hai thuộc tính có thang đo khác nhau, lưu lượng giao thông có giá trị lớn hơn nhiều so với diện tích. Nếu không chuẩn hóa, khoảng cách bị chi phối bởi lưu lượng, làm giảm vai trò của diện tích trong phân cụm. Chuẩn hóa đưa mỗi thuộc tính về trung bình xấp xỉ 0 và độ lệch chuẩn xấp xỉ 1, nhờ đó hai thuộc tính có đóng góp cân bằng hơn khi tính khoảng cách.
```python
import numpy as np

mean = X_original.mean(axis=0)
std = X_original.std(axis=0, ddof=0)
X_scaled = (X_original - mean) / std

def inverse_standardize(Z: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return Z * std + mean
```
## Phần A
### Nguyên lý K means
K means tìm K tâm cụm sao cho tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm của nó là nhỏ nhất. Thuật toán lặp hai bước. Bước gán cụm, mỗi điểm được gán vào cụm có tâm gần nhất. Bước cập nhật tâm cụm, tâm mới của mỗi cụm bằng trung bình của các điểm trong cụm đó. Quá trình lặp dừng khi tâm cụm gần như không đổi hoặc đạt số vòng lặp tối đa.
### Hàm `initialize_centroids`
Hàm khởi tạo chọn ngẫu nhiên K điểm dữ liệu làm tâm cụm ban đầu. `random_state` cố định hạt giống ngẫu nhiên để tái lập kết quả. Khi K lớn hơn số điểm dữ liệu, `replace` cho phép chọn lặp để vẫn tạo đủ số tâm cụm.
```python
import numpy as np
def initialize_centroids(X: np.ndarray, k: int, random_state: int = None) -> np.ndarray:
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    replace = (k > n_samples)
    random_indices = np.random.choice(n_samples, size=k, replace=replace)
    centroids = X[random_indices]
    return centroids
```
### Hàm `compute_distance`
Hàm tính ma trận khoảng cách giữa mọi điểm dữ liệu và mọi centroid. Ma trận có kích thước số điểm nhân số cụm. Mỗi phần tử là khoảng cách Euclidean giữa một điểm và một centroid.
```python
def compute_distance(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    n_samples = X.shape[0]
    k = centroids.shape[0]
    distances = np.zeros((n_samples, k))
    for i in range(n_samples):
        for j in range(k):
            distances[i, j] = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))
    return distances
```
Ma trận khoảng cách là cơ sở để xác định centroid gần nhất của từng điểm dữ liệu.
### Hàm `assign_clusters`
Hàm gán cụm gọi `compute_distance` để nhận ma trận khoảng cách. Mỗi điểm được gán vào cụm có khoảng cách nhỏ nhất, thông qua `argmin` theo trục centroid.
```python
def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = compute_distance(X, centroids)
    labels = np.argmin(distances, axis=1)
    return labels
```
Kết quả `labels` có chiều dài bằng số điểm dữ liệu. Mỗi phần tử là chỉ số cụm của điểm tương ứng.
### Hàm `update_centroids`
Hàm cập nhật centroid tính trung bình của các điểm thuộc từng cụm. Nếu một cụm rỗng, centroid mới được đặt bằng một điểm dữ liệu ngẫu nhiên để thuật toán tiếp tục chạy ổn định.
```python
def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    n_features = X.shape[1]
    new_centroids = np.zeros((k, n_features))
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
        else:
            new_centroids[i] = X[np.random.choice(X.shape[0])]
    return new_centroids
```
Việc lấy trung bình theo cụm là nguyên nhân khiến centroid di chuyển về vị trí đại diện cho cụm, đồng thời làm giảm WCSS qua các vòng lặp.
### Hàm `compute_wcss`
Hàm tính WCSS bằng tổng bình phương sai khác giữa các điểm trong cụm và centroid của cụm. Với cụm i, phần đóng góp vào WCSS là tổng của (x trừ centroid i) bình phương, cộng trên mọi điểm thuộc cụm i.
```python
def compute_wcss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    wcss = 0.0
    k = centroids.shape[0]
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            wcss += np.sum((cluster_points - centroids[i]) ** 2)
    return wcss
```
WCSS vừa là thước đo độ chặt của các cụm vừa là giá trị cần thiết cho phương pháp Elbow ở phần B.
### Hàm `kmeans_clustering`
Hàm này kết hợp toàn bộ các bước của K means. Khởi tạo centroid, lặp gán cụm, tính WCSS, cập nhật centroid. Mỗi vòng lặp lưu WCSS vào `wcss_history`. Điều kiện dừng dựa trên độ dịch chuyển tổng hợp của centroid, được tính bằng căn bậc hai của tổng bình phương chênh lệch giữa centroid cũ và centroid mới.
```python
from typing import Tuple, List
def kmeans_clustering(
    X: np.ndarray,
    k: int,
    max_iters: int = 100,
    random_state: int = None,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    centroids = initialize_centroids(X, k, random_state)
    wcss_history: List[float] = []
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        wcss = compute_wcss(X, labels, centroids)
        wcss_history.append(wcss)
        new_centroids = update_centroids(X, labels, k)
        centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        if centroid_shift < tol:
            break
        centroids = new_centroids
    return labels, centroids, wcss_history
```
`labels` là nhãn cụm cuối cùng của từng điểm. `centroids` là tọa độ centroid cuối cùng trong không gian chuẩn hóa. `wcss_history` là chuỗi WCSS theo vòng lặp.
### Chạy K means với K bằng 3
Phần A chạy thuật toán với K bằng 3 trên dữ liệu chuẩn hóa `X_scaled`.
```python
k = 3
labels_custom, centroids_custom, wcss_history = kmeans_clustering(
    X_scaled, k=k, max_iters=100, random_state=42
)
```
`centroids_custom` nằm trong không gian chuẩn hóa. Để diễn giải centroid theo đơn vị gốc, centroid được chuyển ngược về không gian ban đầu bằng hàm `inverse_standardize`.
```python
centroids_original = inverse_standardize(centroids_custom, mean, std)
```
## Phần B
### Tính WCSS cho K từ 1 đến 9
Phương pháp Elbow yêu cầu WCSS tương ứng với nhiều giá trị K. Với mỗi K, thuật toán K means được chạy đến khi dừng, sau đó lấy WCSS cuối cùng.
```python
k_range = range(1, 10)
wcss_values = []
all_results = {}
for k_val in k_range:
    labels, centroids, wcss_hist = kmeans_clustering(
        X_scaled, k=k_val, max_iters=100, random_state=42
    )
    final_wcss = wcss_hist[-1]
    wcss_values.append(final_wcss)
    all_results[k_val] = {
        "labels": labels,
        "centroids": centroids,
        "wcss": final_wcss,
    }
```
`wcss_values` là dãy WCSS theo K, dùng để vẽ đường cong Elbow. `all_results` lưu lại nhãn cụm, centroid và WCSS của từng K để phục vụ việc trực quan hóa và phân tích.
### Vẽ biểu đồ Elbow
Biểu đồ Elbow biểu diễn WCSS theo K. Khi K tăng, WCSS giảm do các cụm được chia nhỏ hơn. Vị trí tối ưu thường được nhận diện tại vùng mà mức giảm WCSS bắt đầu giảm chậm hơn.
```python
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(k_range, wcss_values, "bo-", linewidth=2, markersize=10, label="WCSS")
plt.scatter([3], [wcss_values[2]], s=300, c="red", marker="*", label="K=3")
plt.xlabel("Số lượng clusters K")
plt.ylabel("WCSS")
plt.title("Elbow Method để xác định K tối ưu")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```
## Kết luận
Phần A triển khai đầy đủ K means thông qua các hàm khởi tạo centroid, tính khoảng cách, gán cụm, cập nhật centroid, tính WCSS và vòng lặp dừng theo ngưỡng `tol`. Phần B chạy K từ 1 đến 9 để lấy WCSS và vẽ Elbow. Báo cáo tập trung vào việc triển khai K means bằng numpy và sử dụng WCSS để xây dựng đường cong Elbow.
