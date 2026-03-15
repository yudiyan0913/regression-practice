# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% jupyter={"source_hidden": true}
import numpy as np
import matplotlib.pyplot as plt


# %%
def read_data(n, m):
    data = np.loadtxt('Linear Regression_n=2,m=50_data.txt', delimiter = ',')
    x = data[:, 0:n]
    x[:, 0] /= 1000
    # You can apply normalization to X and y, setting them all in [0,1]
    y = data[:, n]
    y /= 1000
    return x, y


# %%
def show_graph(x, y, w, b):
    # only when n = 1
    x_extract = np.array([np.min(x), np.max(x)])
    y_extract = w * x_extract + b
    plt.scatter(x, y, marker = 'x', c = 'r')
    plt.plot(x_extract, y_extract, 'b-')
    plt.title("Housing prices")
    plt.xlabel("Size (1000 sqft)")
    plt.ylabel("Price (in 1000s of dollars)")
    plt.show()


# %%
def compute_cost(x, y, w, b, n, m):
    cost = 0
    y_hat = x @ w + b
    for i in range(m):
        cost += 1/2 * (y_hat[i] - y[i]) ** 2
    cost /= m
    return cost


# %%
def compute_gradient_w(x, y, w, b, n, m):
    ans = np.zeros(n)
    y_hat = x @ w + b
    for i in range(m):
        for j in range(n):
            ans[j] += (y_hat[i] - y[i]) * x[i][j]
    ans /= m
    return ans


# %% jupyter={"source_hidden": true}
def compute_gradient_b(x, y, w, b, n, m):
    ans = 0
    y_hat = x @ w + b
    for i in range(m):
        ans += (y_hat[i] - y[i])
    ans /= m
    return ans


# %% jupyter={"source_hidden": true}
def gradient_descent(x, y, w, b, alpha, n, m):
    dj_dw = compute_gradient_w(x, y, w, b, n, m)
    dj_db = compute_gradient_b(x, y, w, b, n, m)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    return w, b


# %%
def predict(x, w, b):
    return x @ w + b


# %%
n = 2
m = 50
x_train, y_train = read_data(n, m)
w = np.zeros(n)
b = 0
alpha = 0.01
iteration = 100000

# %% jupyter={"outputs_hidden": true}
for i in range(iteration):
    w, b = gradient_descent(x_train, y_train, w, b, alpha, n, m)
    if (i + 1) % 5000 == 0:
        print(f"Iteration times: {i+1}, w = {w}, b = {b}, cost = {compute_cost(x_train, y_train, w, b, n, m)}")
print(f"End of Gradient Descent. w = {w}, b = {b}, cost = {compute_cost(x_train, y_train, w, b, n, m)}")
