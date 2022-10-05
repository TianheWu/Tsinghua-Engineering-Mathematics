import numpy as np
import matplotlib.pyplot as plt


def open_file(path):
    ret_list = []
    with open(path, 'r') as listFile:
        for line in listFile:
            x, y = line.split('\t')
            ret_list.append(float(y))
    return ret_list

def compare(pred_path, true_path):
    pred_list = open_file(pred_path)
    true_list = open_file(true_path)
    diff_sum = 0
    for i in range(len(pred_list)):
        if abs(pred_list[i] - true_list[i]) > 0.5e-12:
            print("i = {}".format(i))
            diff_sum += abs(pred_list[i] - true_list[i])
    return diff_sum


def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range

def visual(x, y):
    x = normalization(np.array(x))
    y = normalization(np.array(y))
    plt.xlabel('N')
    plt.ylabel('Time')
    plt.scatter(x, y)
    parameter = np.polyfit(x, y, 1)
    y2 = parameter[0] * x + parameter[1]
    plt.plot(x, y2, color='r')
    plt.savefig('./time.jpg')
    plt.show()

if __name__ == "__main__":
    pred_path = "./results/dp8736_results.txt"
    true_path = "./results/true_results.txt"
    # x = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    # y = [5, 56, 552, 5451, 54818, 532882]
    # visual(x, y)
    print(compare(pred_path, true_path))
