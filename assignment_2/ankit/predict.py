from torch.autograd import Variable
import torch
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
sns.set()

win = tk.Tk()
win.resizable(width=0, height=0)
win.title("Fibonnachi")
n_label = ttk.Label(win, text="Enter the n:")
n_label.grid(row=0, column=0, sticky=tk.W)
n_value = tk.IntVar()
n_value.set(10)
n_entrybox = ttk.Entry(win, width=14, textvariable=n_value)
n_entrybox.grid(row=0, column=1)
a_label = ttk.Label(win, text="a:")
a_label.grid(row=1, column=0, sticky=tk.W)
a_value = tk.IntVar()
a_value.set(1)
a_entrybox = ttk.Entry(win, width=14, textvariable=a_value)
a_entrybox.grid(row=1, column=1)
b_label = ttk.Label(win, text="b:")
b_label.grid(row=2, column=0, sticky=tk.W)
b_value = tk.IntVar()
b_value.set(2)
b_entrybox = ttk.Entry(
    win, width=14, textvariable=b_value)
b_entrybox.grid(row=2, column=1)
n_out = 1
n = 0
A, B = [], []
for i in range(1, 10):
    for j in range(i, 10):
        n += 1
        A.append(i)
        B.append(j)
A, B = np.array(A), np.array(B)
for i in range(n):
    if A[i] > B[i]:
        A[i], B[i] = B[i], A[i]
data = pd.DataFrame({})
data["A"] = A
data["B"] = B
X = np.array(data)
y = np.array(data["A"]+data["B"]).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("X", X[:5], "\n\n", "y", y[:5])
dtype = torch.FloatTensor
N, Din, Dout = n, 2, n_out
x = Variable(torch.Tensor(X_train).type(dtype), requires_grad=False)
x_test = Variable(torch.Tensor(X_test).type(dtype), requires_grad=False)
y = Variable(torch.Tensor(y_train).type(dtype), requires_grad=False)
y_test = Variable(torch.Tensor(y_test).type(dtype), requires_grad=False)
w = Variable(torch.randn(Din, Dout).type(dtype), requires_grad=True)
learning_rate = 0.00001
train_loss_list = []
test_loss_list = []
for t in range(1000):
    y_pred = x.mm(w)
    y_pred_test = x_test.mm(w)
    loss = (y_pred - y).pow(2).sum()
    loss_test = (y_pred_test-y_test).pow(2).sum()
    train_loss_list.append(loss.data)
    test_loss_list.append(loss_test)
    loss.backward()
    w.data -= learning_rate * w.grad.data
    w.grad.data.zero_()
plt.plot(train_loss_list, label="train loss")
plt.plot(test_loss_list, label="test_loss")
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.legend()
plt.show()


def sum_new(a, b, w):
    return round(float(torch.Tensor([[a, b]]).mm(w)[0][0].data))


def fibo(w, a, b, n):
    if n == 1:
        return a
    a_sum_b = sum_new(a, b, w)
    return fibo(w, b, a_sum_b, n-1)


def main():
    a, b, n = a_value.get(), b_value.get(), n_value.get()
    for i in range(1, n+1):
        print(fibo(w, a, b, i), end=" ")
    print("\n")


submit_button = tk.Button(
    win, text="Generate", command=main)
submit_button.grid(row=3, columnspan=3)
win.mainloop()
