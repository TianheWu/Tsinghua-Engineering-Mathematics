import math
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from math import sqrt


e = 0.000000001


def solve_equations(A, b):
    x = np.linalg.inv(A).dot(b)
    # print("x: ", x.flatten())
    return x


def l2_loss(x1, x2):
    return sqrt(mean_squared_error(x1, x2))


def ShowMatrix(matrix):
    row = len(matrix)
    for i in range(row):
        print(matrix[i][:])


def ColMaxGaussMethod(matrix, b):
    print("Column GaussEL===================================")
    """
    列主元高斯消元法
    """
    row = len(matrix)
    col = len(matrix[0])
    x = [0] * col
    for k in range(row - 1):
        max_number = 0
        index = -1
        for i in range(k, row):
            if math.fabs(matrix[i][k]) > math.fabs(max_number):
                max_number = matrix[i][k]
                index = i
        if math.fabs(max_number) <= e:
            print("WARNING! Please recheck the formula and identify it can be solved!")
            return None
        if index != k:
            for i in range(col):
                matrix[k][i], matrix[index][i] = matrix[index][i], matrix[k][i]
            b[k], b[index] = b[index], b[k]
        for j in range(k + 1, row):
            coeff = matrix[j][k] / matrix[k][k]
            for i in range(col):
                matrix[j][i] = matrix[j][i] - coeff * matrix[k][i]
            b[j] = b[j] - coeff * b[k]
    
        print("Step {}, Gauss matrix: ".format(k + 1))
        ShowMatrix(matrix)
        print("Step {} b: ".format(k + 1))
        print(b)
        print("========================================================================================")

    x[len(x) - 1] = b[col - 1] / matrix[col - 1][col - 1]
    for i in range(row - 2, -1, -1):
        for j in range(i, col-1):
            b[i] = b[i] - matrix[i][j + 1] * x[j + 1]
        x[i] = b[i] / matrix[i][i]
    return x


def Cholesky(matrix, b):
    print("Cholesky decomposition==================")
    w = matrix.shape[0]
    G = np.zeros((w, w), dtype=np.float64)
    for i in range(w):
        G[i, i] = (matrix[i, i] - np.dot(G[i,:i], G[i,:i].T)) ** 0.5
        for j in range(i + 1, w):
            G[j, i] = (matrix[j, i] - np.dot(G[j,:i], G[i,:i].T)) / G[i, i]
    # print("L")
    # print(G)
    # print("L-T")
    # print(G.T)

    # print(G.shape)
    y = solve_equations(G, b=b)
    x = solve_equations(G.T, y)
    # print("x")
    # print(x)
    return np.squeeze(x, axis=1)


def generate_matrix(n=20):
    matrix = np.zeros([n, n], np.float64)
    for i in range(n):
        matrix[i, i] = 3
        if i >= 1:
            matrix[i, i - 1] = -1
        if i <= n-2:
            matrix[i, i + 1] = -1
        if i != n // 2 - 1 and i != n // 2:
            matrix[i, n - 1 - i] = 1 / 2
    b = np.mat(np.ones([n, 1]) * 1.5)
    b[[0, n - 1]] = 2.5
    b[[n // 2 - 1, n // 2]] = 1.0
    matrix = np.mat(matrix)
    return matrix, b


def Crout(matrix, b):
    print("Crout decomposition==================")
    w = matrix.shape[0]
    T = np.zeros((w, w))
    M = np.zeros((w, w))
    for i in range(w):
        M[i, i] = 1
    for i in range(w):
        for j in range(i, w):
            T[j, i] = matrix[j, i] - np.dot(T[j, :i], M[:i, i])
        for j in range(i + 1, w):
            M[i, j] = (matrix[i, j] - np.dot(T[i, :i], M[:i, j])) / T[i, i]
    print("T:")
    print(T)
    print("M:")
    print(M)
    y = solve_equations(T, b=b)
    x = solve_equations(M, y)
    return x



if __name__ == '__main__':
    # test_b = [[-1, -3, 0, 12, 1, 5, 0, -2, 0, 1],
    #     [-3, 2, 10, -3, -4, 8, -1, 0, -5, 0],
    #     [3, 1, -1, -8, 7, 29, 7, 2, 0, 0],
    #     [-2, 9, 11, 3, 15, -2, 6, 2, 1, 4],
    #     [-1, 5, 2, -3, 2, -1, -9, 5, -2, 12],
    #     [4, 9, -8, 11, 1, -2, 5, 6, -1, 0],
    #     [0, 5, -2, 6, -9, 23, 4, 0, 9, 17],
    #     [-8, 10, -4, 9, 12, 15, 2, -1, 0, 2],
    #     [-4, 0, 2, -7, 1, 3, 21, 0, 8, -11],
    #     [4, 0, -2, 3, -13, 2, -3, 32, 0, -1]]
    # b = [16, -14, 31, 49, -1, 30, 20, 24, 24, 2]

    # print('测试列主元高斯消元法')
    # result = ColMaxGaussMethod(test_b, b)
    # print('result=', result)

    # label = [1, -1, 0, 1, 2, 0, 3, 1, -1, 2]
    # print(l2_loss(np.array(label), np.array(result)))

    # matrix = np.array([
    #     [3,-1, 0, 0, 0, 0, 0, 0, 0, 0.5],
    #     [-1, 3, -1, 0, 0, 0, 0, 0, 0.5, 0],
    #     [0, -1, 3, -1, 0, 0, 0, 0.5, 0, 0],
    #     [0, 0, -1, 3, -1, 0, 0.5, 0, 0, 0],
    #     [0, 0, 0, -1, 3, -1, 0, 0, 0, 0],
    #     [0, 0, 0, 0, -1, 3, -1, 0, 0, 0],
    #     [0, 0, 0, 0.5, 0, -1, 3, -1, 0, 0],
    #     [0, 0, 0.5, 0, 0, 0, -1, 3, -1, 0],
    #     [0, 0.5, 0, 0, 0, 0, 0, -1, 3, -1],
    #     [0.5, 0, 0, 0, 0, 0, 0, 0, -1, 3]
    # ], dtype=np.float64)
    # b = np.array([[2.5], [1.5], [1.5], [1.5], [1.0], [1.0], [1.5], [1.5], [1.5], [2.5]], dtype=np.float64)
    # x = Cholesky(matrix, b)
    
    # print(l2_loss(x, solve_equations(matrix, b)))

    # A, b = generate_matrix(n=50)
    # start_time = time.time()
    # x = Cholesky(A, b)
    # end_time = time.time()
    # print("Time: {}ms".format((end_time - start_time) * 1000))
    # print(A)
    # print(b)

    # matrix = np.zeros([10,10])
    # for i in range(10):
    #     matrix[i, i] = 4
    #     if i >= 1:
    #         matrix[i, i - 1] = -1
    #     if i <= 10 - 2:
    #         matrix[i, i + 1] = -1
    # b = np.mat([7, 5, -13, 2, 6, -12, 14, -4, 5, -5], dtype=np.float64).reshape(-1, 1)
    # x = Crout(matrix=matrix, b=b)
    # print(x)

    # x = np.squeeze(x, axis=1)
    # print(x[0][0])

    x = np.array([2, 1, -3, 1.10588622e-17, 1, -2,  3, -2.77555756e-16, 1, -1], dtype=np.float64)
    label = np.array([2, 1, -3, 0, 1, -2, 3, 0, 1, -1], dtype=np.float64)
    # print(label.shape)
    print(l2_loss(x, label))
    