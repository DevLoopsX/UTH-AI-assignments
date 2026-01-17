# Báo Cáo Bài Tập KNN - Dự Đoán Salary Từ Experience

## Tổng Quan Bài Tập

Bài tập này tập trung vào việc ứng dụng thuật toán K-Nearest Neighbors (KNN) để giải quyết bài toán hồi quy, cụ thể là dự đoán mức lương (Salary) dựa trên số năm kinh nghiệm (Experience) của nhân viên. Bài tập được chia thành hai phần chính: phần đầu tiên yêu cầu tự xây dựng hàm KNN từ đầu để hiểu rõ nguyên lý hoạt động của thuật toán, phần thứ hai so sánh kết quả với thư viện Sklearn để xác minh tính chính xác của implementation.

## Dữ Liệu Đề Bài

Tập dữ liệu bao gồm 14 mẫu quan sát với hai thuộc tính. Thuộc tính đầu vào là Experience (số năm kinh nghiệm) có giá trị từ 1.0 đến 7.5 năm với khoảng cách đều 0.5 năm. Thuộc tính đầu ra là Salary (mức lương tính bằng nghìn đô la) với giá trị dao động từ 0.0 đến 101.0. Đặc điểm quan trọng của dữ liệu là các giá trị Experience dưới 3.0 năm có Salary bằng 0, trong khi các giá trị từ 3.0 năm trở lên có Salary tăng dần theo kinh nghiệm, thể hiện mối quan hệ tương quan dương giữa hai biến.

Cấu trúc dữ liệu được tổ chức dưới dạng DataFrame với hai cột, trong đó cột Experience chứa 14 giá trị float đại diện cho số năm làm việc, và cột Salary chứa 14 giá trị float tương ứng đại diện cho mức lương. Dữ liệu này được sử dụng làm tập huấn luyện cho mô hình KNN, và mục tiêu là dự đoán Salary khi Experience có giá trị 6.3 năm.

## Phần 1: Implementation Thuật Toán KNN

### Nguyên Lý Hoạt Động KNN

Thuật toán K-Nearest Neighbors hoạt động dựa trên nguyên lý đơn giản nhưng hiệu quả: các điểm dữ liệu gần nhau trong không gian đặc trưng có xu hướng có giá trị đầu ra tương tự nhau. Trong bài toán hồi quy, KNN dự đoán giá trị cho một điểm mới bằng cách tính trung bình các giá trị của K điểm gần nhất trong tập huấn luyện.

Quá trình hoạt động của KNN được chia thành năm bước tuần tự. Bước đầu tiên là tính toán khoảng cách từ điểm cần dự đoán đến tất cả các điểm trong tập huấn luyện. Bước thứ hai sắp xếp các khoảng cách này theo thứ tự tăng dần để xác định điểm nào gần nhất. Bước thứ ba chọn K điểm có khoảng cách nhỏ nhất, gọi là K neighbors. Bước thứ tư trích xuất giá trị đầu ra của K neighbors này. Bước cuối cùng tính toán giá trị dự đoán bằng cách lấy trung bình cộng của các giá trị đầu ra từ K neighbors.

### Hàm euclidean_distance

Hàm euclidean_distance thực hiện việc tính toán khoảng cách Euclidean giữa hai điểm trong không gian một chiều. Đây là công thức cơ bản và quan trọng nhất trong thuật toán KNN vì nó quyết định điểm nào được coi là "gần" điểm cần dự đoán.

```python
def euclidean_distance(x1: float, x2: float) -> float:
    return abs(x1 - x2)
```

Hàm này nhận vào hai tham số x1 và x2 đều có kiểu float, đại diện cho hai giá trị cần tính khoảng cách. Do bài toán chỉ có một chiều dữ liệu (Experience), công thức khoảng cách Euclidean được đơn giản hóa thành giá trị tuyệt đối của hiệu hai số. Hàm trả về một giá trị float là khoảng cách giữa hai điểm. Trong không gian nhiều chiều, công thức sẽ phức tạp hơn với căn bậc hai của tổng bình phương các hiệu, nhưng với một chiều, công thức tối giản thành |x1 - x2|.

Việc sử dụng hàm abs() đảm bảo khoảng cách luôn có giá trị dương, đây là tính chất bắt buộc của một metric khoảng cách hợp lệ. Hàm này được gọi nhiều lần trong quá trình tính toán, cụ thể là n lần với n là số lượng điểm trong tập huấn luyện, do đó độ phức tạp thời gian của hàm này là O(1) - hằng số, đảm bảo hiệu suất tốt.

### Hàm knn_predictor

Hàm knn_predictor là hàm chính thực hiện toàn bộ logic của thuật toán KNN. Đây là hàm cốt lõi nhất trong implementation, tích hợp tất cả các bước từ tính khoảng cách đến đưa ra dự đoán cuối cùng.

```python
def knn_predictor(X_train: np.ndarray, y_train: np.ndarray,
                  x_query: float, k: int = 3) -> Tuple[float, List[Tuple[float, float, float]]]:
```

Hàm nhận bốn tham số đầu vào. X_train là mảng numpy chứa tất cả giá trị Experience trong tập huấn luyện, có shape (n_samples,) với n_samples = 14 trong trường hợp này. y_train là mảng numpy chứa các giá trị Salary tương ứng, cũng có shape (n_samples,). x_query là giá trị Experience cần dự đoán Salary, trong bài toán này là 6.3. k là số lượng neighbors gần nhất sử dụng cho dự đoán, mặc định là 3, đây là hyperparameter có thể điều chỉnh để tối ưu hóa mô hình.

Hàm trả về một tuple gồm hai phần tử. Phần tử đầu tiên là prediction, một giá trị float là Salary được dự đoán, được tính bằng trung bình của k neighbors. Phần tử thứ hai là neighbors_info, một danh sách các tuple, mỗi tuple chứa ba giá trị (experience, salary, distance) của một neighbor, giúp người dùng hiểu rõ quá trình dự đoán dựa trên những điểm nào.

Flow hoạt động của hàm được thực hiện qua năm bước chi tiết. Bước 1 khởi tạo danh sách distances rỗng, sau đó duyệt qua từng điểm trong X_train bằng vòng lặp for với chỉ số i từ 0 đến len(X_train)-1. Với mỗi điểm, hàm gọi euclidean_distance(x_query, X_train[i]) để tính khoảng cách từ điểm cần dự đoán đến điểm thứ i, sau đó append tuple (i, dist) vào danh sách distances, trong đó i là chỉ số của điểm và dist là khoảng cách. Kết quả của bước này là một danh sách chứa 14 tuple, mỗi tuple lưu trữ chỉ số và khoảng cách của một điểm trong tập huấn luyện.

Bước 2 thực hiện sắp xếp danh sách distances theo khoảng cách tăng dần bằng phương thức sort() với key=lambda x: x[1]. Lambda function này chỉ định rằng việc sắp xếp dựa trên phần tử thứ hai của tuple (chính là khoảng cách). Sau khi sắp xếp, các điểm gần nhất với x_query sẽ nằm ở đầu danh sách, điều này rất quan trọng cho bước tiếp theo. Độ phức tạp thời gian của bước này là O(n log n) với n là số lượng điểm trong tập huấn luyện.

Bước 3 sử dụng list comprehension để trích xuất k chỉ số đầu tiên từ danh sách distances đã sắp xếp. Câu lệnh k*nearest_indices = [idx for idx, * in distances[:k]] tạo ra một danh sách chứa chỉ số của k điểm gần nhất. Ký hiệu \_ trong unpacking tuple biểu thị rằng giá trị khoảng cách không cần sử dụng ở bước này. Kết quả là một danh sách có độ dài k, mỗi phần tử là chỉ số của một neighbor trong X_train và y_train.

Bước 4 sử dụng list comprehension khác để lấy giá trị Salary của các neighbors. Câu lệnh k_nearest_salaries = [y_train[i] for i in k_nearest_indices] duyệt qua các chỉ số trong k_nearest_indices và trích xuất giá trị tương ứng từ y_train. Kết quả là một danh sách chứa k giá trị Salary của các neighbors gần nhất, đây là dữ liệu cốt lõi để tính toán dự đoán cuối cùng.

Bước 5 tính toán giá trị dự đoán bằng hàm np.mean(k_nearest_salaries). Hàm này tính trung bình cộng của tất cả k giá trị Salary trong danh sách, đây chính là nguyên lý cơ bản của KNN trong bài toán hồi quy. Công thức toán học là prediction = (Σ salary_i) / k với i từ 1 đến k. Giá trị này được gán vào biến prediction và sẽ là kết quả trả về chính của hàm.

Phần cuối cùng của hàm tạo neighbors_info để cung cấp thông tin chi tiết về các neighbors. Sử dụng list comprehension kết hợp với enumerate, hàm tạo ra danh sách các tuple (X_train[i], y_train[i], distances[j][1]) với i là chỉ số của neighbor và j là vị trí trong danh sách distances đã sắp xếp. Mỗi tuple chứa ba thông tin: giá trị Experience của neighbor, giá trị Salary của neighbor, và khoảng cách từ neighbor đến điểm query. Thông tin này rất hữu ích cho việc debug, phân tích và giải thích kết quả dự đoán cho người dùng.

### Độ Phức Tạp Thuật Toán

Độ phức tạp thời gian của thuật toán KNN được phân tích theo từng bước. Bước tính khoảng cách có độ phức tạp O(n) vì cần duyệt qua tất cả n điểm trong tập huấn luyện. Bước sắp xếp có độ phức tạp O(n log n) do sử dụng thuật toán sắp xếp hiệu quả của Python. Bước chọn k neighbors, trích xuất giá trị và tính trung bình đều có độ phức tạp O(k), với k thường là hằng số nhỏ. Tổng hợp lại, độ phức tạp thời gian của toàn bộ thuật toán là O(n log n), trong đó n là kích thước tập huấn luyện.

Độ phức tạp không gian chủ yếu đến từ danh sách distances lưu trữ n tuple, chiếm O(n) không gian. Các biến khác như k_nearest_indices, k_nearest_salaries và neighbors_info chỉ chiếm O(k) không gian, thường là nhỏ so với n. Do đó, độ phức tạp không gian tổng thể là O(n).

## Phần 2: So Sánh Với Sklearn

### Sử Dụng KNeighborsRegressor

Phần này sử dụng class KNeighborsRegressor từ module sklearn.neighbors để so sánh kết quả với implementation tự viết. Sklearn là thư viện machine learning phổ biến và đáng tin cậy, do đó việc so sánh giúp xác minh tính đúng đắn của code.

Quy trình sử dụng Sklearn bắt đầu với việc reshape dữ liệu. Sklearn yêu cầu dữ liệu đầu vào phải có shape (n_samples, n_features), do đó X_train cần được reshape từ (14,) thành (14, 1) bằng phương thức reshape(-1, 1). Tương tự, x_query cũng được chuyển thành mảng 2D với shape (1, 1). Việc này đảm bảo định dạng dữ liệu tương thích với API của Sklearn.

Tiếp theo, khởi tạo model bằng câu lệnh knn_sklearn = KNeighborsRegressor(n_neighbors=k). Tham số n_neighbors tương đương với tham số k trong hàm tự viết, xác định số lượng neighbors sử dụng cho dự đoán. Model sau đó được huấn luyện bằng phương thức fit(X_train_sklearn, y_train), trong đó Sklearn lưu trữ toàn bộ tập huấn luyện để sử dụng sau này.

Dự đoán được thực hiện bằng phương thức predict(x_query_sklearn), trả về một mảng numpy chứa giá trị dự đoán. Do chỉ dự đoán một điểm, kết quả được trích xuất bằng [0] để lấy giá trị float. Ngoài ra, phương thức kneighbors(x_query_sklearn) cung cấp thông tin chi tiết về distances và indices của các neighbors, giúp phân tích sâu hơn về quá trình dự đoán.

### Kết Quả So Sánh

Kết quả so sánh cho thấy implementation tự viết và Sklearn cho ra giá trị hoàn toàn giống nhau với k=3. Cả hai đều dự đoán Salary là 94.00k khi Experience = 6.3. Sự khác biệt giữa hai phương pháp là 0.000000k, xác nhận tính chính xác tuyệt đối của implementation.

Với các giá trị k khác nhau (k=1, 3, 5, 7), kết quả cũng hoàn toàn trùng khớp. Điều này chứng minh rằng logic thuật toán được implement chính xác, từ việc tính khoảng cách, sắp xếp, chọn neighbors đến tính trung bình. Việc so sánh này không chỉ xác minh tính đúng đắn của code mà còn tăng độ tin cậy khi áp dụng thuật toán vào thực tế.

Thông tin về neighbors từ cả hai phương pháp cũng giống nhau. Với k=3 và x_query=6.3, cả hai đều xác định ba neighbors gần nhất là các điểm có Experience 6.0, 6.5 và 7.0, với khoảng cách lần lượt là 0.3, 0.2 và 0.7. Điều này cho thấy không chỉ kết quả cuối cùng mà cả quá trình trung gian đều nhất quán giữa hai implementation.

## Kết Quả Dự Đoán Chi Tiết

### Phân Tích Với k=3

Với giá trị k=3 (mặc định), hàm knn_predictor dự đoán Salary là 94.00k khi Experience = 6.3 năm. Ba neighbors gần nhất được xác định như sau:

Neighbor thứ nhất có Experience 6.5 năm, Salary 91.00k và khoảng cách 0.2. Đây là neighbor gần nhất với điểm query, có ảnh hưởng lớn nhất đến kết quả dự đoán. Khoảng cách chỉ 0.2 năm cho thấy điểm này rất tương đồng với điểm cần dự đoán.

Neighbor thứ hai có Experience 6.0 năm, Salary 93.00k và khoảng cách 0.3. Đây là neighbor gần thứ hai, cũng có ảnh hưởng đáng kể. Giá trị Salary của neighbor này cao hơn neighbor đầu tiên, tạo ra sự cân bằng trong dự đoán.

Neighbor thứ ba có Experience 7.0 năm, Salary 98.00k và khoảng cách 0.7. Đây là neighbor xa nhất trong ba neighbors được chọn, có ảnh hưởng nhỏ nhất nhưng vẫn đóng góp vào kết quả cuối cùng. Giá trị Salary cao của neighbor này kéo kết quả dự đoán lên một chút.

Giá trị dự đoán được tính bằng trung bình cộng của ba Salaries: (91.00 + 93.00 + 98.00) / 3 = 282.00 / 3 = 94.00k. Tuy nhiên, cần lưu ý rằng trong thực tế, do có thể có sự làm tròn số hoặc thứ tự sắp xếp khác nhau khi có nhiều điểm cùng khoảng cách, kết quả có thể dao động nhẹ xung quanh giá trị này.

### Phân Tích Với Các Giá Trị k Khác

Khi thay đổi giá trị k, kết quả dự đoán cũng thay đổi, phản ánh ảnh hưởng của hyperparameter này đến model.

Với k=1, model chỉ sử dụng neighbor gần nhất (Experience 6.5, Salary 91.00k), dẫn đến dự đoán là 91.00k. Đây là trường hợp đơn giản nhất, kết quả hoàn toàn phụ thuộc vào một điểm duy nhất. Ưu điểm là model rất nhạy với dữ liệu local, nhược điểm là dễ bị ảnh hưởng bởi nhiễu và outliers.

Với k=5, model sử dụng năm neighbors gần nhất, bao gồm các điểm Experience 6.5, 6.0, 7.0, 5.5 và 7.5. Dự đoán sẽ là trung bình của năm Salaries tương ứng. Việc tăng k làm cho model mượt mà hơn, ít bị ảnh hưởng bởi nhiễu, nhưng có thể làm mất đi các đặc điểm chi tiết của dữ liệu.

Với k=7, model sử dụng bảy neighbors, càng tăng tính tổng quát. Tuy nhiên, nếu k quá lớn so với kích thước tập dữ liệu (14 điểm), model có thể bị underfitting, không nắm bắt được mối quan hệ thực sự giữa Experience và Salary.

## Đánh Giá Hiệu Suất Model

### Leave-One-Out Cross-Validation

Để đánh giá hiệu suất tổng quát của model, notebook sử dụng phương pháp Leave-One-Out Cross-Validation (LOOCV). Đây là kỹ thuật cross-validation đặc biệt phù hợp với tập dữ liệu nhỏ như trong bài toán này.

LOOCV hoạt động bằng cách lần lượt loại bỏ từng điểm dữ liệu, sử dụng n-1 điểm còn lại để huấn luyện model, sau đó dự đoán điểm bị loại bỏ. Quá trình này được lặp lại n lần với n là tổng số điểm (14 trong trường hợp này). Mỗi lần lặp tạo ra một cặp giá trị (actual, predicted), và cuối cùng các metrics được tính toán trên toàn bộ n cặp giá trị này.

Với mỗi fold trong LOOCV, tập huấn luyện có 13 điểm và tập test có 1 điểm. Model KNN được khởi tạo với n_neighbors = min(k, len(train_index)) để đảm bảo k không vượt quá số lượng điểm trong tập huấn luyện. Sau khi fit model trên 13 điểm, model dự đoán điểm còn lại và so sánh với giá trị thực tế.

### Các Metrics Đánh Giá

Hiệu suất của model được đánh giá qua ba metrics chính: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE) và R² Score.

MAE đo lường trung bình độ lệch tuyệt đối giữa giá trị dự đoán và giá trị thực tế. Công thức là MAE = (1/n) Σ|y_true - y_pred|. MAE có đơn vị giống với biến đầu ra (nghìn đô la trong trường hợp này), dễ hiểu và diễn giải. Giá trị MAE càng nhỏ, model càng chính xác.

RMSE đo lường căn bậc hai của trung bình bình phương sai số. Công thức là RMSE = √[(1/n) Σ(y_true - y_pred)²]. RMSE nhạy cảm hơn với outliers so với MAE do sử dụng bình phương sai số. RMSE cũng có đơn vị giống biến đầu ra, giá trị càng nhỏ càng tốt.

R² Score (Coefficient of Determination) đo lường tỷ lệ phương sai của biến đầu ra được giải thích bởi model. Công thức là R² = 1 - (SS_res / SS_tot), trong đó SS_res là tổng bình phương residuals và SS_tot là tổng bình phương sai lệch so với mean. R² có giá trị từ -∞ đến 1, với 1 là hoàn hảo, 0 nghĩa là model không tốt hơn việc dự đoán bằng mean, và giá trị âm nghĩa là model tệ hơn việc dự đoán bằng mean.

Kết quả đánh giá cho thấy model với k=3 có hiệu suất cân bằng, không quá overfitting (k quá nhỏ) hay underfitting (k quá lớn). Việc so sánh các metrics giữa các giá trị k khác nhau giúp lựa chọn hyperparameter tối ưu cho bài toán.

## Trực Quan Hóa

Notebook bao gồm nhiều biểu đồ trực quan hóa để giúp hiểu rõ hơn về dữ liệu và kết quả dự đoán.

Biểu đồ scatter plot đầu tiên hiển thị mối quan hệ giữa Experience và Salary. Mỗi điểm dữ liệu được vẽ với màu xanh, kích thước lớn và viền đen để dễ quan sát. Đường thẳng đứng đỏ đứt nét tại Experience = 6.3 đánh dấu vị trí cần dự đoán. Biểu đồ này cho thấy rõ xu hướng tăng của Salary theo Experience, đặc biệt sau mốc 3.0 năm.

Biểu đồ subplot 2x2 hiển thị kết quả dự đoán với bốn giá trị k khác nhau (1, 3, 5, 7). Mỗi subplot bao gồm các điểm training data màu xanh nhạt, k neighbors được highlight màu cam với viền đỏ đậm, điểm dự đoán được đánh dấu bằng ngôi sao đỏ lớn, và các đường nét đứt màu xanh lá kết nối điểm dự đoán với các neighbors. Visualization này giúp thấy rõ ảnh hưởng của k đến việc chọn neighbors và giá trị dự đoán cuối cùng.

Các đường kết nối giữa điểm dự đoán và neighbors giúp visualize quá trình "lấy trung bình" trong KNN. Độ dài của các đường này phản ánh khoảng cách từ điểm query đến từng neighbor, giúp hiểu tại sao những neighbor gần hơn có ảnh hưởng quan trọng hơn (mặc dù KNN đơn giản sử dụng trung bình không가중).

## Ưu Điểm và Hạn Chế

### Ưu Điểm Của KNN

KNN là thuật toán đơn giản, dễ hiểu và dễ implement, không yêu cầu kiến thức toán học phức tạp. Thuật toán không có giai đoạn training phức tạp, chỉ cần lưu trữ dữ liệu và thực hiện tính toán khi dự đoán. KNN hoạt động tốt với dữ liệu có mối quan hệ phi tuyến, không giả định về phân phối dữ liệu. Thuật toán linh hoạt, có thể áp dụng cho cả classification và regression với việc điều chỉnh nhỏ trong bước tính toán kết quả cuối.

### Hạn Chế Của KNN

Độ phức tạp tính toán cao khi dự đoán, phải tính khoảng cách đến tất cả điểm trong tập huấn luyện, dẫn đến O(n) cho mỗi prediction. Với tập dữ liệu lớn, thời gian dự đoán có thể chậm đáng kể. KNN nhạy cảm với scale của features; nếu các features có đơn vị khác nhau (ví dụ: meter và kilometer), feature có giá trị lớn hơn sẽ dominance trong tính toán khoảng cách. Do đó, cần normalization hoặc standardization trước khi áp dụng KNN.

Thuật toán cũng nhạy cảm với nhiễu và outliers, đặc biệt khi k nhỏ. Một outlier có thể ảnh hưởng lớn đến dự đoán nếu nằm trong k neighbors. Việc chọn k phù hợp là một thách thức; k quá nhỏ dẫn đến overfitting, k quá lớn dẫn đến underfitting. Cần sử dụng cross-validation để tìm k tối ưu.

KNN không phù hợp với dữ liệu có nhiều chiều (curse of dimensionality). Khi số chiều tăng, khái niệm "gần" trở nên mơ hồ vì tất cả các điểm đều xa nhau tương đối. Cuối cùng, KNN yêu cầu lưu trữ toàn bộ tập training data, chiếm nhiều bộ nhớ với dữ liệu lớn.

## Kết Luận

Bài tập đã hoàn thành thành công hai mục tiêu chính. Thứ nhất, implementation hàm knn_predictor từ đầu với logic rõ ràng, từng bước được giải thích chi tiết, giúp hiểu sâu nguyên lý hoạt động của KNN. Thứ hai, so sánh với Sklearn xác nhận tính chính xác tuyệt đối của implementation, với sai số bằng 0 giữa hai phương pháp.

Kết quả dự đoán cho Experience = 6.3 là Salary 94.00k (với k=3) là hợp lý, nằm trong khoảng của các neighbors gần nhất và phù hợp với xu hướng tăng của dữ liệu. Ba neighbors được sử dụng có Experience 6.0, 6.5 và 7.0 với Salaries 93.00k, 91.00k và 98.00k, cho thấy vùng dữ liệu xung quanh điểm query có tính ổn định tốt.

Việc phân tích với nhiều giá trị k khác nhau và sử dụng LOOCV để đánh giá hiệu suất giúp có cái nhìn tổng quan về model. Trực quan hóa bằng các biểu đồ làm rõ ràng hơn quá trình dự đoán và ảnh hưởng của hyperparameter k.

Thuật toán KNN tuy đơn giản nhưng hiệu quả với bài toán này do tập dữ liệu nhỏ và mối quan hệ giữa Experience và Salary tương đối rõ ràng. Với dữ liệu lớn hơn hoặc phức tạp hơn, có thể cần xem xét các cải tiến như KD-Tree, Ball Tree để tăng tốc tìm kiếm neighbors, hoặc weighted KNN để neighbors gần có trọng số cao hơn.

## File Structure

```
BT-KNN/
├── knn_salary_prediction.ipynb    # Notebook chính chứa code và phân tích
└── README.md                       # Báo cáo chi tiết này
```

## Cách Sử Dụng

1. Mở file knn_salary_prediction.ipynb bằng Jupyter Notebook hoặc JupyterLab
2. Chạy từng cell theo thứ tự từ trên xuống dưới
3. Quan sát kết quả output và các biểu đồ trực quan
4. Có thể thay đổi giá trị k hoặc x_query để thử nghiệm với các trường hợp khác

## Thư Viện Yêu Cầu

- numpy: Xử lý mảng và tính toán số học
- pandas: Quản lý dữ liệu dạng bảng
- matplotlib: Vẽ biểu đồ và trực quan hóa
- sklearn: So sánh với implementation chuẩn và đánh giá model

Cài đặt bằng lệnh:

```bash
pip install numpy pandas matplotlib scikit-learn
```
