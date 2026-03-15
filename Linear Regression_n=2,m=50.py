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

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def read_data():
    data = np.loadtxt('Linear Regression_n=2,m=50_data.txt', delimiter = ',')
    x = data[:, 0:2]
    x[:, 0] /= 1000
    y = data[:, 2]
    y /= 1000
    return x, y


# %%
def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    y_hat = x @ w + b
    for i in range(m):
        cost += 1/2 * (y_hat[i] - y[i]) ** 2
    cost /= m
    return cost


# %%
def compute_gradient_w(x, y, w, b):
    m = len(x)
    ans = np.zeros(2)
    y_hat = x @ w + b
    for i in range(m):
        for j in range(2):
            ans[j] += (y_hat[i] - y[i]) * x[i][j]
    ans /= m
    return ans


# %%
def compute_gradient_b(x, y, w, b):
    m = len(x)
    ans = 0
    y_hat = x @ w + b
    for i in range(m):
        ans += (y_hat[i] - y[i])
    ans /= m
    return ans


# %%
def gradient_descent(x, y, w, b, alpha):
    dj_dw = compute_gradient_w(x, y, w, b)
    dj_db = compute_gradient_b(x, y, w, b)
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    return w, b


# %%
x_train, y_train = read_data()
w = np.zeros(2)
b = 0
alpha = 0.01
iteration = 100000

# %%
for i in range(iteration):
    w, b = gradient_descent(x_train, y_train, w, b, alpha)
    if (i + 1) % 1000 == 0:
        print(f"Iteration times: {i+1}, w = {w}, b = {b}, cost = {compute_cost(x_train, y_train, w, b)}")
print(f"End of Gradient Descent. w = {w}, b = {b}, cost = {compute_cost(x_train, y_train, w, b)}")

# %%
