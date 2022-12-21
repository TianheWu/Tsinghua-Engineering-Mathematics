import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def function(p, x):
    a0, a1, a2, a3, a4 = p
    return a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4

def power_function(p, x):
    a0, a1 = p
    return a0 * x ** a1

def error(p, x, y):
    return power_function(p, x) - y

def lsm(x, y):
    p = np.array([1, 0.5])
    para = leastsq(error, p, args=(x, y))
    
    y_fitted = power_function(para[0], x)

    error_sum = 0
    for i in range(x.shape[0]):
        error_sum += (y_fitted[i] - y[i]) ** 2
    print(error_sum)

    plt.scatter(x, y, label='Original Point')
    plt.scatter(x, y_fitted, label='Fitted Point')
    
    for i in range(x.shape[0]):
        if i == 0:
            plt.plot([x[i], x[i]], [y[i], y_fitted[i]], color='yellow', label="Error")
        else:
            plt.plot([x[i], x[i]], [y[i], y_fitted[i]], color='yellow')

    plt.legend()
    plt.savefig("./error_power.jpg", dpi=600, bbox_inches='tight')
    plt.show()

    # x_list = np.linspace(0, 55, 10000)
    # y_fitted = power_function(para[0], x_list)
    # plt.plot(x_list, y_fitted, 'y', label ='Fitted curve')
    # plt.legend()
    # plt.savefig("./fitted_power.jpg", dpi=600, bbox_inches='tight')
    # plt.show()
    # print(para[0])

if __name__=='__main__':
    x = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
    y = np.array([0, 1.27, 2.16, 2.86, 3.44, 3.87, 4.15, 4.37, 4.51, 4.58, 4.02, 4.64])
    lsm(x, y)
    # plt.scatter(x, y)
    # plt.title("Carbon content versus time")
    # plt.xlabel("t (min)")
    # plt.ylabel("y (1e-4)")
    # plt.savefig("./visual_data.jpg", dpi=600, bbox_inches='tight')
    # plt.show()


