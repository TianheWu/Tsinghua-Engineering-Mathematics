import numpy as np
from sympy import *

np.set_printoptions(formatter={'float': '{: 0.7f}'.format})


def lagrange_interpolation(x_list, y_list, x):
    # the base function of lagrange
    l = np.zeros(shape=(x_list.shape[0]))
    pred_val = 0
    for i in range(x_list.shape[0]):
        l[i] = 1.0
        for j in range(x_list.shape[0]):
            if j != i:
                # calculate the base function
                l[i] *= (x - x_list[j]) / (x_list[i] - x_list[j])
        pred_val += l[i] * y_list[i]
    print(pred_val)
    return pred_val, l


def hermite_interpolation(x_list, y_list, x):
    pass


def cal_derivative(x, f, s, e, table):
    """
    Recursively calculate the difference quotient
    """
    if e - s == 1:
        table[e - 1][e - s - 1] = (f[e] - f[s]) / (x[e] - x[s])
        return table[e - 1][e - s - 1]
    pre = cal_derivative(x, f, s + 1, e, table)
    last = cal_derivative(x, f, s, e - 1, table)
    table[e - 1][e - s - 1] = (pre - last) / (x[e] - x[s])
    return table[e - 1][e - s - 1]


def newton_interpolation(x, f):
    table = np.ones([x.shape[0] - 1, x.shape[0] - 1])
    cal_derivative(x, f, 0, x.shape[0] - 1, table)

    print(table)

    X = symbols("x")
    y = f[0]
    for i in range(x.shape[0] - 1):
        temp = 1
        for j in range(i + 1):
            temp = temp * (X - x[j])
        temp = table[i, i] * temp 
        y = y + temp
    
    print(y.evalf(subs={X:1.8}))
    print("N(x)=", y)
    return y


if __name__ == '__main__':
    # x_list = np.array([0.5, 2, 2.5, 4])
    # y_list = np.array([2, 0.5, 0.4, 0.25])
    # x = 3
    # pred_val, l = lagrange_interpolation(x_list=x_list, y_list=y_list, x=x)
    # print(pred_val)

    x = np.array([1, 2, 3, 4, 5, 6, 7])
    f = np.array([0.368, 0.135, 0.050, 0.018, 0.007, 0.002, 0.001])
    # y = newton_interpolation(x, f)
    pred_val, l = lagrange_interpolation(x_list=x, y_list=f, x=1.8)
    
    

    

