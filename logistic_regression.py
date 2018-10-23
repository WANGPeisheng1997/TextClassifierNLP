import math
import numpy as np
import random
import time

# x : [ [x1,x2,...,xd], [x1,x2,...,xd], ..., [x1,x2,...,xd] ]
# y : [y1,y2,y3,...,yN]
# weight : [b, w1, w2, ..., wd]
def logistic_regression(sample_x, sample_y, learning_rate, epoch, batch_size):
    N = len(sample_x)  # sample count
    dim = len(sample_x[0])  # dimension
    weight = np.zeros(dim + 1)
    extended_x = np.hstack((np.ones((N, 1)), sample_x))

    for i in range(epoch):
        random_index = random.sample(range(N), N)
        iterations = [random_index[i : i + batch_size] for i in range(0, N - batch_size + 1, batch_size)]
        for iterate in iterations:
            delta_weight = np.zeros(dim + 1)
            mat_x = [extended_x[index] for index in iterate]
            mat_y = [sample_y[index] for index in iterate]
            #mat_x is (batch_size * dim+1) and weight is (1 * dim+1)

            pr = [logistic_sigmoid(h) for h in np.transpose(np.matmul(mat_x, np.transpose(weight)))]

            for i in range(batch_size):
                delta_weight += (mat_y[i]-pr[i]) * mat_x[i] * learning_rate / batch_size
            weight += delta_weight
    return weight

    # sum = 0
    # for (y, ye) in zip(sample_y, yn):
    #     loss_single = y * math.log2(ye) + (1 - y) * math.log2(1 - ye)
    #     sum += loss_single
    # loss = (-1) / N * sum
    # print("loss: " + str(loss))


def logistic_sigmoid(x):
    e = math.exp(-x)
    return 1/(1+e)