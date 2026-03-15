# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def read_data(n, m):
    data = np.loadtxt('Logistic Regression_n=2,m=26_data.txt', delimiter = ',')
    x = data[:, 0:n]
    x /= 100
    y = data[:, n]
    return x, y

# %%
def show_graph(x, y, w, b):
    # only when n = 2
    pos = y == 1
    neg = y == 0
    x1_extract = np.array([np.min(x[:, 0]), np.max(x[:, 0])])
    if w[1] == 0:
        x2_extract = np.zeros(2)
    else:
        x2_extract = - (w[0] * x1_extract + b) / w[1]
    plt.scatter(x[pos,0] * 100, x[pos,1] * 100, marker = '+', c = 'r')
    plt.scatter(x[neg,0] * 100, x[neg,1] * 100, marker = 'o', c = 'b')
    plt.plot(x1_extract * 100, x2_extract * 100, 'g-')
    plt.title("Exams")
    plt.xlabel("Exam 1 Score")
    plt.ylabel("Exam 2 Score")
    plt.show()

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# %%
def compute_cost(x, y, w, b, n, m):
    cost = 0
    y_hat = sigmoid(x @ w + b)
    for i in range(m):
        if y[i] == 1:
            cost += - np.log(y_hat[i] + 1e-15)
        else:
            cost += - np.log(1 - y_hat[i] + 1e-15)
    cost /= m
    return cost

# %%
def compute_gradient_w(x, y, w, b, n, m):
    ans = np.zeros(n)
    y_hat = sigmoid(x @ w + b)
    for i in range(m):
        for j in range(n):
            ans[j] += (y_hat[i] - y[i]) * x[i][j]
    ans /= m
    return ans

# %%
def compute_gradient_b(x, y, w, b, n, m):
    ans = 0
    y_hat = sigmoid(x @ w + b)
    for i in range(m):
        ans += (y_hat[i] - y[i])
    ans /= m
    return ans

# %%
def gradient_descent(x, y, w, b, alpha, n, m):
    dj_dw = compute_gradient_w(x, y, w, b, n, m)
    dj_db = compute_gradient_b(x, y, w, b, n, m)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    return w, b

# %%
def predict(x, w, b):
    return sigmoid(x / 100 @ w + b)

# %%
n = 2
m = 26
x_train, y_train = read_data(n, m)
w = np.zeros(n)
b = 0
alpha = 0.01
iteration = 100000

# %%
for i in range(iteration):
    w, b = gradient_descent(x_train, y_train, w, b, alpha, n, m)
    if (i + 1) % 10000 == 0:
        print(f"Iteration times: {i+1}, w = {w}, b = {b}, cost = {compute_cost(x_train, y_train, w, b, n, m)}")
        show_graph(x_train, y_train, w, b)
print(f"End of Gradient Descent. w = {w}, b = {b}, cost = {compute_cost(x_train, y_train, w, b, n, m)}")


