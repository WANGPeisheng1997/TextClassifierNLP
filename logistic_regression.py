import math
import numpy as np
# [ [[x1,x2,...,xn],y],[[x1,x2,...,xn],y],[[x1,x2,...,xn],y],... ]
def logistic_regression_iterate(samples, alpha, w): # w = b, w1, w2, ..., wd
    N = len(samples)  # sample count
    dim = len(samples[0][0])  # dimension
    yn = []  # ^y(n)
    deltaw = np.array([0] * (dim + 1))
    for sample in samples:
        x = np.array([1] + sample[0])
        # calculate sig(wTx) = ^ywtn
        pr = logistic_sigmoid(x.dot(w))
        yn.append(pr)

        # calculate xn * (yn - ^ywt(n))
        deltaw = deltaw + (sample[1] - pr) * x
    deltaw = deltaw * alpha / N

    sum = 0
    for i in range(N):
        loss_single = samples[i][1] * math.log2(yn[i]) + (1 - samples[i][1]) * math.log2(1 - yn[i])
        sum = sum + loss_single
    loss = (-1) / N * sum
    print("loss: " + str(loss))

    return w + deltaw


def logistic_sigmoid(x):
    e = math.exp(-x)
    return 1/(1+e)


def logistic_regression(samples, alpha, iterate_times):
    dim = len(samples[0][0])  # dimension
    w0 = np.array([0] * (dim+1))
    w = w0
    for iterate_count in range(iterate_times):
        print("Iterate: " + str(iterate_count))
        w = logistic_regression_iterate(samples, alpha, w)
        print(w)
    print("Finished!")

# arr = [ [[0],0],[[1],0],[[2],1], [[1.8],1], [[5],1] , [[1.4], 0] , [[1.9], 0]]

arr = [
    [[1, 1], 0],
    [[1, 2], 0],
    [[1, 3], 0],
    [[2, 2], 0],
    [[2, 3], 0],
    [[3, 4], 0],
    [[3, 1], 1],
    [[4, 1], 1],
    [[6, 1], 1],
    [[5, 2], 1],
    [[6, 3], 1]
]
logistic_regression(arr, 1, 10000)


# x = np.array([3,4])
# y = np.array([5,6])
# print(x.dot(y))
