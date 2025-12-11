import math

#dataset
dataset = [
    [0.5, 0],
    [1.0, 0],
    [1.5, 0],
    [2.0 , 0],
    [2.5, 1],
    [3.0, 1],
    [3.5, 1],
    [4.0, 1]
]

def get_prediction(m, b, x):
    y = m * x + b
    return 1 / ( 1 + math.exp(- y))

def get_cost(y, y_hat):
    k = len(y)
    total_cost = 0.0
    for yi, y_hat_i in zip(y, y_hat):
        total_cost += - ( yi * math.log(y_hat_i) + (1 - yi) * math.log(1 - y_hat_i) )
    return total_cost / k

def get_gradients(m, b, x, y, y_hat):
    k = len(y)
    dm = (1 / k) * sum((y_hat_i - y_i) * x_i for y_hat_i, y_i, x_i in zip(y_hat, y, x))
    db = (1 /k) * sum(y_hat_i - y_i for y_hat_i, y_i in zip(y_hat, y))
    return dm, db

def get_accuracy(y, y_hat):
    correct_predictions = sum((1 if y_hat_i >= 0.5 else 0) == y_i for y_hat_i, y_i in zip(y_hat, y))
    return correct_predictions / len(y)

m = 1.0
b = -1.0

iterations = 10
learning_rate = 1.0

x = [row[0] for row in dataset]
y = [row[1] for row in dataset]

costs = []

for it in range(iterations):

    y_hat = [get_prediction(m, b, x_i) for x_i in x]
    
    cost = get_cost(y, y_hat)
    accuracy = get_accuracy(y, y_hat)
    dm, db = get_gradients(m, b, x, y, y_hat)
    
    # print(f"Iteration {it + 1}: m = {m:.4f}, b = {b:.4f}, Accuracy = {accuracy:.4f}")
   
    #Update
    m -= learning_rate * dm
    b -= learning_rate * db
    costs.append(cost)

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