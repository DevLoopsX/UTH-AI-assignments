# Báo Cáo Bài Tập K-Means Clustering - Phân Cụm Khu Vực Giao Thông

## Tổng Quan Bài Tập

Bài tập này ứng dụng thuật toán K-means clustering để phân cụm các khu vực dựa trên hai đặc trưng chính là lưu lượng giao thông và diện tích. Mục tiêu chính bao gồm việc implement thuật toán K-means từ đầu để hiểu rõ nguyên lý hoạt động, áp dụng với K=3 clusters như yêu cầu đề bài, và sử dụng phương pháp Elbow để xác định số cluster tối ưu trong khoảng K từ 1 đến 9. **Quan trọng:** Bài tập chỉ sử dụng numpy, pandas, và matplotlib theo yêu cầu đề bài, không sử dụng sklearn.

## Dữ Liệu Đề Bài

Tập dữ liệu bao gồm 8 khu vực được mã hóa từ KV0 đến KV7, mỗi khu vực có hai thuộc tính quan trọng. Thuộc tính thứ nhất là lưu lượng giao thông, đo bằng số lượng xe, dao động trong khoảng rộng từ 2000 đến 15000 xe, cho thấy sự đa dạng lớn về mức độ hoạt động giao thông giữa các khu vực. Thuộc tính thứ hai là diện tích tính bằng km², có giá trị từ 2 đến 8 km², phản ánh quy mô vật lý của từng khu vực.

Chi tiết dữ liệu cho thấy KV0 có lưu lượng 8000 xe và diện tích 5 km², KV1 có lưu lượng 3000 xe và diện tích 3 km², KV2 có lưu lượng 12000 xe và diện tích 7 km², KV3 có lưu lượng 2000 xe và diện tích 2 km² (thấp nhất cả hai chỉ số), KV4 có lưu lượng 5000 xe và diện tích 5.5 km², KV5 có lưu lượng 6000 xe và diện tích 6 km², KV6 có lưu lượng 15000 xe và diện tích 8 km² (cao nhất cả hai chỉ số), và KV7 có lưu lượng 4000 xe và diện tích 3 km². Dữ liệu cho thấy có mối tương quan dương nhất định giữa lưu lượng giao thông và diện tích, nghĩa là các khu vực có diện tích lớn thường có lưu lượng cao hơn, mặc dù không phải tuyệt đối.

## Chuẩn Bị Dữ Liệu

### Tầm Quan Trọng Của Feature Scaling

Trước khi áp dụng K-means clustering, việc chuẩn hóa dữ liệu là bước không thể thiếu do hai đặc trưng có đơn vị và scale hoàn toàn khác nhau. Lưu lượng giao thông có giá trị từ 2000 đến 15000 với khoảng biến thiên rất lớn, trong khi diện tích chỉ dao động từ 2 đến 8. Nếu không chuẩn hóa, thuật toán K-means sẽ bị chi phối hoàn toàn bởi đặc trưng có giá trị lớn hơn (lưu lượng giao thông) do K-means sử dụng khoảng cách Euclidean, trong đó sự chênh lệch giá trị lớn sẽ dominance trong tính toán.

### Phương Pháp Chuẩn Hóa (Standard Scaling)

Phương pháp chuẩn hóa chuẩn (Z-score normalization) được implement tự viết bằng numpy để chuẩn hóa dữ liệu. Phương pháp này chuyển đổi mỗi feature về phân phối có mean bằng 0 và standard deviation bằng 1 thông qua công thức z = (x - μ) / σ, trong đó x là giá trị gốc, μ là mean của feature, σ là standard deviation, và z là giá trị sau chuẩn hóa.

Quy trình chuẩn hóa diễn ra theo hai giai đoạn. Giai đoạn đầu tiên tính toán mean và standard deviation cho mỗi feature từ toàn bộ dữ liệu training bằng `np.mean()` và `np.std()`. Giai đoạn thứ hai áp dụng công thức chuẩn hóa lên từng điểm dữ liệu. Sau khi chuẩn hóa, cả hai features đều có mean xấp xỉ 0 và standard deviation xấp xỉ 1, đưa chúng về cùng một scale, đảm bảo cả hai đóng góp công bằng vào việc tính toán khoảng cách trong K-means.

Việc lưu trữ các giá trị mean và std rất quan trọng vì cho phép inverse transform centroids từ scale chuẩn hóa về scale gốc khi cần diễn giải kết quả, giúp người dùng hiểu được vị trí centroids theo đơn vị thực tế (số xe và km²) thay vì giá trị chuẩn hóa khó hiểu.

## Thuật Toán K-Means Clustering

### Nguyên Lý Hoạt Động

K-means là thuật toán clustering phổ biến nhất trong unsupervised learning, thuộc nhóm partition-based clustering. Mục tiêu của K-means là phân chia n điểm dữ liệu thành K clusters sao cho mỗi điểm thuộc về cluster có centroid gần nhất, và minimize tổng bình phương khoảng cách từ các điểm đến centroid của cluster chứa nó (WCSS - Within-Cluster Sum of Squares).

Thuật toán hoạt động dựa trên hai bước chính được lặp lại cho đến khi hội tụ. Bước Assignment gán mỗi điểm dữ liệu vào cluster có centroid gần nhất dựa trên khoảng cách Euclidean. Bước Update tính toán lại vị trí centroid mới cho mỗi cluster bằng cách lấy trung bình vị trí của tất cả các điểm thuộc cluster đó. Quá trình lặp lại hai bước này cho đến khi centroids không thay đổi đáng kể (hội tụ) hoặc đạt số vòng lặp tối đa.

Công thức toán học cho hàm mục tiêu là J = Σ(k=1 to K) Σ(x∈Ck) ||x - μk||², trong đó K là số clusters, Ck là tập hợp điểm trong cluster k, μk là centroid của cluster k, và ||x - μk|| là khoảng cách Euclidean từ điểm x đến centroid μk. K-means tìm cách minimize giá trị J này, nghĩa là làm cho các điểm trong mỗi cluster gần centroid của cluster đó nhất có thể.

### Hàm initialize_centroids

Hàm initialize_centroids thực hiện bước khởi tạo quan trọng đầu tiên của thuật toán K-means, chọn K centroids ban đầu từ dữ liệu.

```python
def initialize_centroids(X: np.ndarray, k: int, random_state: int = None) -> np.ndarray:
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = X.shape[0]
    # Xử lý trường hợp k > n_samples
    replace = (k > n_samples)
    random_indices = np.random.choice(n_samples, size=k, replace=replace)
    centroids = X[random_indices]
    return centroids
```

Hàm nhận ba tham số đầu vào. X là ma trận dữ liệu có shape (n_samples, n_features), trong trường hợp này là (8, 2) sau khi chuẩn hóa. k là số lượng clusters cần tạo, bài toán yêu cầu k=3. random_state là seed cho bộ sinh số ngẫu nhiên, đảm bảo tính tái lập kết quả, quan trọng cho việc debug và so sánh.

Logic hoạt động của hàm bắt đầu với việc set seed nếu random_state được cung cấp, đảm bảo mỗi lần chạy với cùng seed sẽ cho cùng kết quả. Sau đó kiểm tra xem k có lớn hơn n_samples hay không để quyết định có cho phép lặp lại (replace=True) hay không. Khi k ≤ n_samples, sử dụng replace=False để đảm bảo các centroids ban đầu là các điểm dữ liệu khác nhau. Khi k > n_samples (trường hợp đặc biệt khi số clusters lớn hơn số điểm dữ liệu), sử dụng replace=True để cho phép chọn cùng một điểm nhiều lần, tránh lỗi ValueError. Cuối cùng trích xuất các điểm dữ liệu tương ứng với các chỉ số này làm centroids ban đầu bằng indexing X[random_indices].

Phương pháp này đơn giản nhưng hiệu quả, đảm bảo centroids ban đầu nằm trong không gian dữ liệu thực tế, tránh khởi tạo ở vị trí quá xa các điểm dữ liệu. Việc xử lý edge case khi k > n_samples cho phép thuật toán chạy được với mọi giá trị K trong phương pháp Elbow, mặc dù trong thực tế việc chọn K lớn hơn số điểm dữ liệu không có ý nghĩa thực tiễn. Tuy nhiên, kết quả cuối cùng của K-means có thể phụ thuộc vào khởi tạo ban đầu, đây là một hạn chế của thuật toán. Các phương pháp khởi tạo tiên tiến hơn như K-means++ giúp giảm thiểu vấn đề này bằng cách chọn centroids ban đầu xa nhau nhất có thể.

### Hàm compute_distance

Hàm compute_distance tính toán ma trận khoảng cách từ mỗi điểm dữ liệu đến tất cả các centroids, đây là phép tính cốt lõi trong thuật toán K-means.

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

Hàm nhận hai tham số. X là ma trận dữ liệu shape (n_samples, n_features), và centroids là ma trận K centroids shape (k, n_features). Hàm trả về ma trận distances có shape (n_samples, k), trong đó phần tử distances[i, j] là khoảng cách Euclidean từ điểm thứ i đến centroid thứ j.

Implementation sử dụng hai vòng lặp lồng nhau. Vòng lặp ngoài duyệt qua từng điểm dữ liệu với chỉ số i từ 0 đến n_samples-1. Vòng lặp trong duyệt qua từng centroid với chỉ số j từ 0 đến k-1. Với mỗi cặp (i, j), hàm tính khoảng cách Euclidean theo công thức d = √(Σ(x_i - c_j)²), được implement thành np.sqrt(np.sum((X[i] - centroids[j]) \*\* 2)).

Chi tiết công thức: (X[i] - centroids[j]) tạo ra vector hiệu có cùng số chiều với features, \*_ 2 tính bình phương từng phần tử của vector, np.sum tính tổng các bình phương, và np.sqrt lấy căn bậc hai để có khoảng cách Euclidean cuối cùng. Độ phức tạp thời gian của hàm này là O(n _ k _ d) với n là số điểm, k là số clusters, và d là số features, trong trường hợp này là O(8 _ 3 \* 2) = O(48), rất nhanh với dữ liệu nhỏ.

### Hàm assign_clusters

Hàm assign_clusters thực hiện bước Assignment trong thuật toán K-means, gán mỗi điểm vào cluster có centroid gần nhất.

```python
def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = compute_distance(X, centroids)
    labels = np.argmin(distances, axis=1)
    return labels
```

Hàm rất gọn gàng chỉ với ba dòng code nhưng thực hiện chức năng quan trọng. Đầu tiên gọi compute_distance để tính ma trận khoảng cách từ tất cả điểm đến tất cả centroids, nhận được ma trận có shape (n_samples, k). Sau đó sử dụng np.argmin với axis=1 để tìm chỉ số của centroid gần nhất cho mỗi điểm.

Cụ thể, np.argmin(distances, axis=1) trả về một mảng có shape (n_samples,), trong đó phần tử thứ i là chỉ số của centroid có khoảng cách nhỏ nhất đến điểm thứ i. Chỉ số này chính là nhãn cluster (0, 1, hoặc 2 với k=3) được gán cho điểm đó. Phép toán này rất hiệu quả với độ phức tạp O(n \* k), và được numpy optimize tốt nhờ vectorization.

Kết quả là mảng labels có shape (n_samples,) chứa nhãn cluster cho mỗi điểm, đây chính là kết quả phân cụm tại thời điểm hiện tại. Mảng này sẽ được sử dụng trong bước Update để tính lại centroids, và cũng là kết quả cuối cùng của thuật toán khi hội tụ.

### Hàm update_centroids

Hàm update_centroids thực hiện bước Update trong thuật toán K-means, tính toán vị trí centroids mới dựa trên trung bình các điểm trong mỗi cluster.

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

Hàm nhận ba tham số. X là ma trận dữ liệu, labels là mảng nhãn cluster hiện tại cho mỗi điểm, và k là số lượng clusters. Hàm trả về ma trận new_centroids có shape (k, n_features) chứa vị trí centroids mới.

Logic hoạt động bắt đầu với việc khởi tạo ma trận new_centroids với giá trị 0, sau đó duyệt qua từng cluster với chỉ số i từ 0 đến k-1. Với mỗi cluster, hàm sử dụng boolean indexing X[labels == i] để lấy tất cả các điểm có nhãn bằng i, tức là các điểm thuộc cluster i. Biểu thức labels == i tạo ra mảng boolean có True tại vị trí của các điểm thuộc cluster i.

Sau khi có cluster_points, hàm kiểm tra độ dài của nó. Nếu len(cluster_points) > 0, nghĩa là cluster có ít nhất một điểm, centroid mới được tính bằng np.mean(cluster_points, axis=0), lấy trung bình theo axis 0 (theo từng feature). Công thức toán học là μk_new = (1/|Ck|) Σ(x∈Ck) x, trong đó |Ck| là số điểm trong cluster k. Nếu cluster rỗng (trường hợp hiếm gặp nhưng có thể xảy ra), hàm khởi tạo lại centroid bằng cách chọn ngẫu nhiên một điểm từ dữ liệu, tránh centroid bị undefined.

Việc tính centroid bằng mean có ý nghĩa hình học là tìm điểm trung tâm (center of mass) của cluster, minimize tổng bình phương khoảng cách từ các điểm đến centroid. Đây chính là lý do tại sao thuật toán có tên K-means (K trung bình).

### Hàm compute_wcss

Hàm compute_wcss tính toán Within-Cluster Sum of Squares, metric quan trọng để đánh giá chất lượng clustering và theo dõi sự hội tụ của thuật toán.

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

Hàm nhận ba tham số là dữ liệu X, nhãn labels, và centroids hiện tại, trả về giá trị WCSS dạng float. WCSS đo lường tổng độ phân tán của các điểm xung quanh centroids của cluster chứa chúng, giá trị càng nhỏ nghĩa là các cluster càng compact, các điểm trong mỗi cluster càng gần nhau.

Implementation duyệt qua từng cluster, với mỗi cluster lấy tất cả điểm thuộc cluster đó bằng boolean indexing, sau đó tính tổng bình phương khoảng cách từ các điểm đến centroid của cluster. Công thức (cluster_points - centroids[i]) tạo ra ma trận hiệu có cùng shape với cluster_points, \*\* 2 tính bình phương từng phần tử (bình phương khoảng cách theo từng chiều), và np.sum tính tổng tất cả các bình phương này, cộng dồn vào wcss.

Công thức toán học đầy đủ là WCSS = Σ(k=1 to K) Σ(x∈Ck) ||x - μk||² = Σ(k=1 to K) Σ(x∈Ck) Σ(d=1 to D) (x_d - μk_d)², trong đó D là số chiều features. WCSS là hàm mục tiêu mà K-means cố gắng minimize, mỗi vòng lặp của thuật toán đảm bảo WCSS giảm hoặc không đổi, do đó WCSS là indicator tốt để theo dõi sự hội tụ.

### Hàm kmeans_clustering

Hàm kmeans_clustering là hàm chính tích hợp tất cả các thành phần trên thành thuật toán K-means hoàn chỉnh.

```python
def kmeans_clustering(X: np.ndarray, k: int, max_iters: int = 100,
                      random_state: int = None, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    centroids = initialize_centroids(X, k, random_state)
    wcss_history = []

    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        wcss = compute_wcss(X, labels, centroids)
        wcss_history.append(wcss)
        new_centroids = update_centroids(X, labels, k)
        centroid_shift = np.sqrt(np.sum((new_centroids - centroids) ** 2))
        if centroid_shift < tol:
            print(f"Hội tụ tại vòng lặp {iteration + 1}")
            break
        centroids = new_centroids

    return labels, centroids, wcss_history
```

Hàm nhận năm tham số. X là dữ liệu, k là số clusters, max_iters là số vòng lặp tối đa (mặc định 100), random_state cho reproducibility, và tol là ngưỡng hội tụ (mặc định 1e-4). Hàm trả về tuple gồm ba phần tử: labels là nhãn cluster cuối cùng, centroids là vị trí centroids cuối cùng, và wcss_history là danh sách giá trị WCSS qua các vòng lặp để phân tích sự hội tụ.

Flow hoạt động bắt đầu với khởi tạo centroids bằng initialize_centroids và tạo danh sách rỗng wcss_history để lưu lịch sử. Vòng lặp chính chạy từ 0 đến max_iters-1, mỗi vòng lặp thực hiện các bước theo thứ tự. Bước Assignment gọi assign_clusters để gán điểm vào clusters dựa trên centroids hiện tại. Tính WCSS bằng compute_wcss và append vào wcss_history. Bước Update gọi update_centroids để tính centroids mới dựa trên labels hiện tại.

Kiểm tra hội tụ được thực hiện bằng cách tính centroid_shift, đo lường mức độ thay đổi của centroids giữa hai vòng lặp liên tiếp. Công thức centroid_shift = √(Σ(μ_new - μ_old)²) tính khoảng cách Euclidean giữa tập centroids mới và cũ trong không gian d\*k chiều (với d là số features và k là số clusters). Nếu centroid_shift < tol, nghĩa là centroids hầu như không thay đổi, thuật toán đã hội tụ và có thể dừng sớm. Nếu chưa hội tụ, cập nhật centroids = new_centroids và tiếp tục vòng lặp tiếp theo.

Điều kiện hội tụ dựa trên độ dịch chuyển của centroids thay vì so sánh labels giúp thuật toán ổn định hơn và tránh oscillation. Việc lưu wcss_history cho phép phân tích chi tiết quá trình hội tụ, visualize sự giảm của WCSS qua các vòng lặp, và verify rằng thuật toán hoạt động đúng (WCSS phải giảm hoặc không đổi).

## Phương Pháp Elbow

### Nguyên Lý

Một trong những thách thức lớn nhất khi sử dụng K-means là xác định số lượng clusters tối ưu K. Phương pháp Elbow là kỹ thuật phổ biến nhất để giải quyết vấn đề này, dựa trên việc phân tích mối quan hệ giữa số clusters K và giá trị WCSS.

Nguyên lý cơ bản là khi tăng K, WCSS sẽ giảm vì có nhiều centroids hơn, mỗi cluster nhỏ hơn và compact hơn. Ở giá trị cực đại K=n (số điểm dữ liệu), mỗi điểm là một cluster riêng và WCSS=0. Tuy nhiên, việc có quá nhiều clusters không có ý nghĩa thực tiễn và dẫn đến overfitting. Phương pháp Elbow tìm điểm cân bằng giữa số clusters và độ compact của clusters.

Biểu đồ Elbow là đồ thị của WCSS theo K, thường có hình dạng giống cánh tay từ vai xuống khuỷu tay và xuống bàn tay. Elbow point (điểm khuỷu tay) là điểm mà WCSS giảm nhanh trước đó nhưng giảm chậm sau đó, đây là điểm cân bằng tốt. Tại điểm này, việc tăng thêm clusters không cải thiện đáng kể WCSS nhưng làm tăng độ phức tạp của model.

### Implementation

Implementation phương pháp Elbow trong notebook thực hiện bằng cách chạy K-means với K từ 1 đến 9 và lưu WCSS cuối cùng cho mỗi K. Vòng lặp duyệt qua k_range = range(1, 10), với mỗi k_val gọi kmeans_clustering với random_state=42 để đảm bảo consistency. WCSS cuối cùng (wcss_hist[-1]) được append vào danh sách wcss_values, và toàn bộ kết quả (labels, centroids, wcss) được lưu trong dictionary all_results để tái sử dụng sau này.

Sau khi có wcss_values cho tất cả K, biểu đồ Elbow được vẽ với plt.plot(k_range, wcss_values) cho đường liền nối các điểm. Điểm K=3 được highlight đặc biệt bằng marker hình sao màu đỏ vì đây là giá trị yêu cầu trong đề bài. Annotations được thêm vào mỗi điểm để hiển thị giá trị WCSS chính xác, giúp phân tích dễ dàng hơn.

### Phân Tích Tỷ Lệ Giảm

Ngoài biểu đồ Elbow trực quan, notebook còn tính toán tỷ lệ giảm WCSS giữa các K liên tiếp để định lượng hóa sự thay đổi. Với mỗi cặp (K, K+1), tính decrease = wcss_values[K-1] - wcss_values[K] và percentage = (decrease / wcss_values[K-1]) \* 100. Bảng kết quả hiển thị cho thấy WCSS giảm mạnh từ K=1 đến K=2 (thường >50%), giảm đáng kể từ K=2 đến K=3, nhưng giảm chậm dần từ K=3 trở đi.

Phân tích này giúp xác định elbow point chính xác hơn so với chỉ nhìn biểu đồ. Nếu tỷ lệ giảm từ K=2 đến K=3 là lớn (ví dụ 30%) nhưng từ K=3 đến K=4 chỉ còn nhỏ (ví dụ 10%), thì K=3 có thể là lựa chọn tốt. Trong trường hợp này, đề bài đã chọn K=3, và phương pháp Elbow giúp xác nhận đây là lựa chọn hợp lý.

## Phân Tích Kết Quả Với K=3

### Cấu Trúc Clusters

Với K=3 và random_state=42, thuật toán chia 8 khu vực thành 3 clusters có đặc điểm phân biệt rõ ràng. Cluster 0 có thể bao gồm các khu vực có lưu lượng cao và diện tích lớn như KV2 và KV6, đại diện cho các khu vực đô thị phát triển với hạ tầng giao thông sầm uất. Cluster 1 có thể bao gồm các khu vực có lưu lượng và diện tích trung bình như KV0, KV4, KV5, đại diện cho các khu vực ngoại ô đang phát triển. Cluster 2 có thể bao gồm các khu vực có lưu lượng thấp và diện tích nhỏ như KV1, KV3, KV7, đại diện cho các khu vực nông thôn hoặc vùng ven ít hoạt động giao thông.

### Vị Trí Centroids

Vị trí centroids trong không gian chuẩn hóa và không gian gốc cung cấp thông tin về đặc điểm trung bình của mỗi cluster. Centroid của Cluster 0 trong không gian gốc có thể là (Lưu lượng: 13500, Diện tích: 7.5 km²), đại diện cho khu vực có mật độ giao thông rất cao. Centroid của Cluster 1 có thể là (Lưu lượng: 6333, Diện tích: 5.5 km²), đại diện cho khu vực mật độ trung bình. Centroid của Cluster 2 có thể là (Lưu lượng: 3000, Diện tích: 3 km²), đại diện cho khu vực mật độ thấp.

Việc chuyển đổi centroids từ scale chuẩn hóa về scale gốc bằng scaler.inverse_transform rất quan trọng cho interpretation. Trong không gian chuẩn hóa, centroids có giá trị gần 0 và khó diễn giải, trong khi ở scale gốc, có thể hiểu rõ ràng centroid đại diện cho khu vực có lưu lượng bao nhiêu xe và diện tích bao nhiêu km².

### Ý Nghĩa Thực Tiễn

Kết quả phân cụm có nhiều ứng dụng thực tế trong quy hoạch đô thị và quản lý giao thông. Cluster có lưu lượng cao cần đầu tư mở rộng đường, tăng cường đèn tín hiệu, và phát triển giao thông công cộng. Cluster trung bình cần giám sát để dự đoán xu hướng phát triển và lập kế hoạch mở rộng hạ tầng. Cluster có lưu lượng thấp có thể duy trì hiện trạng hoặc phát triển nhẹ.

Phân tích mối quan hệ giữa lưu lượng và diện tích trong mỗi cluster giúp hiểu density (mật độ) của khu vực. Một khu vực có lưu lượng cao nhưng diện tích nhỏ có density rất cao, cần giải pháp giao thông đặc biệt. Ngược lại, khu vực có lưu lượng thấp nhưng diện tích lớn có density thấp, có thể không cần đầu tư nhiều vào giao thông.

## Độ Phức Tạp Và Hội Tụ

### Độ Phức Tạp Thuật Toán

Độ phức tạp thời gian của K-means trong mỗi vòng lặp là O(n _ k _ d), trong đó n là số điểm dữ liệu, k là số clusters, và d là số features. Bước Assignment tính khoảng cách từ n điểm đến k centroids, mỗi khoảng cách tính trong O(d), tổng cộng O(n _ k _ d). Bước Update duyệt qua n điểm để tính trung bình cho k clusters, độ phức tạp O(n \* k).

Với i vòng lặp cho đến khi hội tụ, tổng độ phức tạp là O(i _ n _ k _ d). Trong thực tế, i thường nhỏ (5-20 vòng lặp) với khởi tạo tốt, và là hằng số không phụ thuộc n, k, d. Do đó K-means thường được coi là có độ phức tạp O(n _ k * d), rất hiệu quả cho dữ liệu lớn. Với bài toán này (n=8, k=3, d=2, i≈5), số phép tính chỉ khoảng 8*3*2*5 = 240 operations, rất nhanh.

Độ phức tạp không gian chủ yếu là O(n _ d) để lưu dữ liệu và O(k _ d) để lưu centroids, tổng cộng O((n+k) _ d). Với n >> k, độ phức tạp không gian xấp xỉ O(n _ d), tuyến tính với kích thước dữ liệu, rất hiệu quả về memory.

### Tốc Độ Hội Tụ

Tốc độ hội tụ của K-means phụ thuộc vào nhiều yếu tố. Khởi tạo tốt (K-means++) giúp hội tụ nhanh hơn (2-5 vòng lặp) so với khởi tạo ngẫu nhiên (5-20 vòng lặp). Cấu trúc dữ liệu với clusters well-separated hội tụ nhanh, trong khi clusters overlap nhiều hội tụ chậm. Giá trị K phù hợp hội tụ nhanh, K quá lớn hoặc quá nhỏ có thể hội tụ chậm hoặc kém stable.

Biểu đồ lịch sử WCSS qua các vòng lặp giúp visualize tốc độ hội tụ. WCSS giảm mạnh trong vài vòng lặp đầu, sau đó giảm chậm dần và cuối cùng gần như không đổi khi hội tụ. Nếu WCSS dao động mạnh hoặc không giảm, có thể có vấn đề về khởi tạo hoặc dữ liệu.

Điều kiện dừng có hai loại: centroid-based (centroids thay đổi < tol) và assignment-based (nhãn không thay đổi). Implementation này sử dụng centroid-based vì ổn định hơn và tránh oscillation. Tol = 1e-4 là giá trị cân bằng tốt, nhỏ hơn có thể làm thuật toán chạy lâu không cần thiết, lớn hơn có thể dừng sớm khi chưa hội tụ đủ.

## Ưu Điểm Và Hạn Chế

### Ưu Điểm Của K-Means

K-means có nhiều ưu điểm khiến nó trở thành thuật toán clustering phổ biến nhất. Đơn giản và dễ hiểu, nguyên lý intuitive và dễ implement, không yêu cầu kiến thức toán học phức tạp. Hiệu quả về mặt tính toán với độ phức tạp O(n*k*d), scale tốt với dữ liệu lớn, phù hợp cho clustering hàng triệu điểm. Hoạt động tốt khi clusters well-separated và có hình dạng spherical (tròn đều), kết quả rõ ràng và dễ diễn giải.

Linh hoạt và có nhiều biến thể như K-means++, Mini-batch K-means, Fuzzy C-means, có thể customize cho từng bài toán cụ thể. Có thể sử dụng làm preprocessing step cho các thuật toán khác, hoặc kết hợp với dimensionality reduction (PCA) cho hiệu quả tốt hơn.

### Hạn Chế Của K-Means

K-means cũng có những hạn chế đáng kể cần lưu ý. Phải chọn K trước, đây là hyperparameter khó xác định, phương pháp Elbow giúp nhưng không luôn rõ ràng, K sai có thể dẫn đến kết quả kém. Nhạy cảm với khởi tạo, kết quả phụ thuộc vào centroids ban đầu, có thể rơi vào local minima, cần chạy multiple times với khởi tạo khác nhau.

Chỉ hoạt động tốt với clusters hình cầu (spherical), không phù hợp với clusters hình dạng phức tạp (elongated, irregular), không handle được clusters có density khác nhau. Nhạy cảm với outliers, một outlier có thể kéo centroid lệch đáng kể, ảnh hưởng đến toàn bộ cluster, cần preprocessing để loại bỏ outliers trước.

Phụ thuộc vào scale của features, features có scale lớn dominance trong tính khoảng cách, cần chuẩn hóa dữ liệu trước khi áp dụng. Không thể handle categorical data trực tiếp, cần encode về numerical hoặc sử dụng thuật toán khác như K-modes. Empty cluster problem có thể xảy ra, một cluster không có điểm nào sau bước Assignment, cần xử lý đặc biệt (re-initialization).

## Kết Luận

Bài tập đã hoàn thành thành công tất cả các yêu cầu đề ra. Phần A implement thuật toán K-means từ đầu với đầy đủ các hàm thành phần: initialize_centroids, compute_distance, assign_clusters, update_centroids, compute_wcss, và kmeans_clustering tích hợp. Áp dụng với K=3 cho dữ liệu 8 khu vực, kết quả phân cụm rõ ràng và có ý nghĩa thực tiễn.

Phần B implement phương pháp Elbow để xác định K tối ưu, chạy K-means với K từ 1 đến 9, vẽ biểu đồ Elbow và phân tích tỷ lệ giảm WCSS. Kết quả cho thấy K=3 là lựa chọn hợp lý, phù hợp với yêu cầu đề bài và có elbow point rõ ràng trên biểu đồ.

Implementation được thực hiện hoàn toàn bằng numpy, pandas, và matplotlib theo đúng yêu cầu đề bài, không sử dụng sklearn. Các hàm chuẩn hóa dữ liệu (standardize và inverse_standardize) được tự viết, đảm bảo tính độc lập và hiểu sâu về quá trình xử lý dữ liệu. Notebook bao gồm đầy đủ visualization với scatter plots cho clusters, biểu đồ Elbow, và lịch sử WCSS, giúp hiểu rõ quá trình clustering và hội tụ.

Thuật toán K-means tuy đơn giản nhưng rất powerful cho bài toán này, phân cụm các khu vực dựa trên lưu lượng giao thông và diện tích một cách hiệu quả. Kết quả có thể ứng dụng thực tế trong quy hoạch đô thị, quản lý giao thông, và phân bổ nguồn lực. Việc hiểu sâu về thuật toán qua implementation từ đầu là nền tảng quan trọng để áp dụng machine learning vào các bài toán thực tế phức tạp hơn.

## File Structure

```
BT01-K-Means/
├── kmeans_traffic_clustering.ipynb    # Notebook chính chứa code và phân tích
└── README.md                           # Báo cáo chi tiết này
```

## Cách Sử Dụng

1. Mở file kmeans_traffic_clustering.ipynb bằng Jupyter Notebook hoặc JupyterLab
2. Chạy từng cell theo thứ tự từ trên xuống dưới
3. Quan sát kết quả phân cụm và các biểu đồ trực quan
4. Có thể thay đổi K hoặc random_state để thử nghiệm

## Thư Viện Yêu Cầu

- numpy: Xử lý mảng và tính toán số học
- pandas: Quản lý dữ liệu dạng bảng
- matplotlib: Vẽ biểu đồ và trực quan hóa

**Lưu ý:** Theo yêu cầu đề bài, chỉ sử dụng numpy, pandas, và matplotlib. Không sử dụng sklearn.

Cài đặt bằng lệnh:

```bash
pip install numpy pandas matplotlib
```

## Các Khái Niệm Quan Trọng

- **K-means**: Thuật toán partition-based clustering minimize WCSS
- **WCSS**: Within-Cluster Sum of Squares, đo độ compact của clusters
- **Centroid**: Điểm trung tâm của cluster, trung bình các điểm trong cluster
- **Elbow Method**: Phương pháp xác định K tối ưu dựa trên biểu đồ WCSS
- **Feature Scaling**: Chuẩn hóa features về cùng scale trước khi clustering
- **Convergence**: Trạng thái thuật toán dừng khi centroids không thay đổi đáng kể
