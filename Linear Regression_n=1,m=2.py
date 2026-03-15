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
def show_graph(x, y, w, b):
    x_extract = np.array([min(x), max(x)])
    y_extract = w * x_extract + b
    plt.scatter(x, y, marker = 'x', c = 'r')
    plt.plot(x_extract, y_extract, 'b-')
    plt.title("Housing prices")
    plt.xlabel("Size (1000 sqft)")
    plt.ylabel("Price (in 1000s of dollars)")
    plt.show()


# %%
def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    y_hat = w * x + b
    for i in range(m):
        cost += 1/2 * (y_hat[i] - y[i]) ** 2
    cost /= m
    return cost


# %%
def compute_gradient_w(x, y, w, b):
    m = len(x)
    ans = 0
    y_hat = w * x + b
    for i in range(m):
        ans += (y_hat[i] - y[i]) * x[i]
    ans /= m
    return ans


# %%
def compute_gradient_b(x, y, w, b):
    m = len(x)
    ans = 0
    y_hat = w * x + b
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
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
w = 0
b = 0
alpha = 0.01
iteration = 10000

# %%
for i in range(iteration):
    w, b = gradient_descent(x_train, y_train, w, b, alpha)
    if (i + 1) % 1000 == 0:
        print(f"Iteration times: {i+1}, w = {w}, b = {b}, cost = {compute_cost(x_train, y_train, w, b)}")
        show_graph(x_train, y_train, w, b)
print(f"End of Gradient Descent. w = {w}, b = {b}, cost = {compute_cost(x_train, y_train, w, b)}")

# %%
