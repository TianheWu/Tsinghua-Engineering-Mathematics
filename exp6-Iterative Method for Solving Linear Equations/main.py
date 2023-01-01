import numpy as np
import time
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


def spectral_radius(matrix):
    a, b = np.linalg.eig(matrix)
    return np.max(np.abs(a))


def l2_loss(x1, x2):
    return sqrt(mean_squared_error(x1, x2))


def generate_matrix(n=20):
    matrix = np.zeros([n, n])
    for i in range(n):
        matrix[i, i] = 3
        if i >= 1:
            matrix[i, i - 1] = -1
        if i <= n-2:
            matrix[i, i + 1] = -1
        if i != n // 2 - 1 and i != n // 2:
            matrix[i, n - 1 - i] = 1 / 2
    b = np.ones([n, 1]) * 1.5
    b[[0, n - 1]] = 2.5
    b[[n // 2 - 1, n // 2]] = 1.0
    # matrix = np.mat(matrix)
    x = np.zeros([n, 1])
    label = np.ones([n, 1])
    return matrix, b, x, label


def Jacobi(A, x, b, error):
    # print("Jacobi==================")
    count = 0
    L = np.array(np.tril(A, -1))
    U = np.array(np.triu(A, 1))
    D_inv = np.diag(1 / np.diag(A))

    error_list = []
    count_list = []
    # B = -D_inv.dot(L + U)
    
    # sr = spectral_radius(B)
    # print("spectral radius: {}, converage speed: {}".format(sr, -np.log(sr)))

    while True:
        xnew = x
        x = D_inv.dot(b - L.dot(x) - U.dot(x))
        error_list.append(abs(x - xnew).max())
        count_list.append(count)
        if abs(x - xnew).max() < error:
            break
        count += 1
        # print("x{}: {}".format(i, x.flatten()))
    
    error_list = np.array(error_list)
    count_list = np.array(count_list)

    plt.plot(count_list, error_list, label="Jacobi")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("./jacobi.jpg", dpi=600, bbox_inches='tight')
    plt.show()

    return x, count


def Gauss(A, x, b, error):
    # print("Gauss==================")
    count = 0

    L = np.array(np.tril(A, -1))
    U = np.array(np.triu(A, 1))
    D = np.array(np.diag(np.diag(A)))

    error_list = []
    count_list = []

    # B = -np.linalg.inv(D + L).dot(U)

    # sr = spectral_radius(B)
    # print("spectral radius: {}, converage speed: {}".format(sr, -np.log(sr)))

    while True:
        xnew = x
        x = np.linalg.inv(D + L).dot(b - U.dot(x))
        error_list.append(abs(x - xnew).max())
        count_list.append(count)
        if abs(x - xnew).max() < error:
            break
        count = count + 1
    
    error_list = np.array(error_list)
    count_list = np.array(count_list)

    plt.plot(count_list, error_list, label="Gauss-Seidel")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("./gauss.jpg", dpi=600, bbox_inches='tight')
    plt.show()

    return x, count


def SOR(A, x, b, w, error):
    # print("SOR==================")
    count = 0

    L = np.array(np.tril(A, -1))
    U = np.array(np.triu(A, 1))
    D = np.array(np.diag(np.diag(A)))

    error_list = []
    count_list = []

    while True:
        xnew = x
        x = np.linalg.inv(D + w * L).dot(w * b + ((1 - w) * D - w * U).dot(x))
        error_list.append(abs(x - xnew).max())
        count_list.append(count)
        if abs(x - xnew).max() < error:
            break
        count = count + 1
    
    error_list = np.array(error_list)
    count_list = np.array(count_list)

    plt.plot(count_list, error_list, label="SOR")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.legend()
    plt.savefig("./SOR.jpg", dpi=600, bbox_inches='tight')
    plt.show()
    
    return x, count


def jacobi2(A, x, b, label, error):
    count = 0
    n = A.shape[0]
    
    error_list = []
    count_list = []

    while True:
        # xnew = x
        for i in range(n):
            temp = 0
            if i == 0:
                temp = x[1] * A[0][1] + x[n - 1] * A[0][n - 1]
            elif i == n - 1:
                temp = x[n - 2] * A[n - 1][n - 2] + x[0] * A[n - 1][0]
            elif i == n // 2 or i == n // 2 - 1:
                temp = x[i - 1] * A[i][i - 1] + x[i + 1] * A[i][i + 1]
            else:
                temp = x[i - 1] * A[i][i - 1] + x[i + 1] * A[i][i + 1] + x[n - 1 - i] * A[i][n - 1 - i]
            x[i] = (b[i] - temp) / A[i][i]

        if abs(x - label).max() < error:
            break
    
    #     count += 1

    #     error_list.append(abs(x - label).max())
    #     count_list.append(count)
    
    # error_list = np.array(error_list)
    # count_list = np.array(count_list)

    # plt.plot(count_list, error_list, label="Jacobi2")
    # plt.xlabel("Iteration")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.savefig("./jacobi2.jpg", dpi=600, bbox_inches='tight')
    # plt.show()

    return x, count


def gauss2(A, x, b, label, error):
    count = 0
    n = A.shape[0]

    x2 = x.copy()
    
    error_list = []
    count_list = []

    while True:
        # xnew = x
        for i in range(n):
            temp = 0
            if i == 0:             
                temp = x[1] * A[0, 1] + x[n - 1] * A[0, n - 1]
            elif i == n - 1:
                temp = x2[n - 2] * A[n - 1, n - 2] + x[0] * A[n - 1, 0]
            elif i == n // 2 or i == n // 2 - 1:
                temp = x2[i - 1] * A[i, i - 1] + x[i + 1] * A[i, i + 1]
            else:                           
                if n - 1 - i > i:
                    temp = x2[i - 1] * A[i, i - 1] + x[i + 1] * A[i, i + 1] + x[n - 1 - i] * A[i, n - 1 - i]
                else:
                    temp = x2[i-1] * A[i, i - 1] + x[i + 1] * A[i, i + 1] + x2[n - 1 - i] * A[i, n - 1 - i]
            x2[i] = (b[i] - temp) / A[i,i]
        x = x2.copy()

        if abs(x - label).max() < error:
            break
    
    #     count += 1

    #     error_list.append(abs(x - label).max())
    #     count_list.append(count)
    
    # error_list = np.array(error_list)
    # count_list = np.array(count_list)

    # plt.plot(count_list, error_list, label="gauss2")
    # plt.xlabel("Iteration")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.savefig("./gauss2.jpg", dpi=600, bbox_inches='tight')
    # plt.show()

    return x, count


def SOR2(A, x, b, label, w, error):
    count = 0
    n = A.shape[0]

    x2 = x.copy()
    
    error_list = []
    count_list = []

    while True:
        # xnew = x
        for i in range(n):
            temp = 0
            if i == 0:             
                temp = x[1] * A[0, 1] + x[n - 1] * A[0, n - 1]
            elif i == n - 1:
                temp = x2[n - 2] * A[n - 1, n - 2] + x[0] * A[n - 1, 0]
            elif i == n // 2 or i == n // 2 - 1:
                temp = x2[i - 1] * A[i, i - 1] + x[i + 1] * A[i, i + 1]
            else:                           
                if n - 1 - i > i:
                    temp = x2[i - 1] * A[i, i - 1] + x[i + 1] * A[i, i + 1] + x[n - 1 - i] * A[i, n - 1 - i]
                else:
                    temp = x2[i-1] * A[i, i - 1] + x[i + 1] * A[i, i + 1] + x2[n - 1 - i] * A[i, n - 1 - i]
            x2[i] = (1 - w) * x[i] + w * (b[i] - temp) / A[i, i]
        x = x2.copy()

        if abs(x - label).max() < error:
            break
    
        count += 1

    #     error_list.append(abs(x - label).max())
    #     count_list.append(count)
    
    # error_list = np.array(error_list)
    # count_list = np.array(count_list)

    # plt.plot(count_list, error_list, label="SOR2")
    # plt.xlabel("Iteration")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.savefig("./SOR2.jpg", dpi=600, bbox_inches='tight')
    # plt.show()

    return x, count


if __name__ == "__main__":
    A, b, x, label = generate_matrix(n=100)
    print("generate sucessfully!")
    x = x.flatten()
    b = b.flatten()
    label = label.flatten()

    start_time = time.time()
    # ans, count = jacobi2(A, x, b, label=label, error=1e-5)
    # ans, count = gauss2(A, x, b, label=label, error=1e-5)
    ans, count = SOR2(A, x, b, label=label, w=1.3, error=1e-5)
    end_time = time.time()

    # ans = np.array(ans)
    # ans = ans.flatten()

    # print("counts: {}".format(count))
    print("Time: {}ms".format((end_time - start_time) * 1000))

    # rmse = l2_loss(ans, label)
    # print("rmse: {}".format(rmse))

