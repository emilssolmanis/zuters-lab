from math import exp
from numpy import *

def read_table(filename):
    with open(filename) as f:
        return array([[float(i) for i in line.split()] for line in f])


def run(samples, weights, g):
    biased_samples = hstack((samples, array([[1] for _ in samples])))
    return 1 / (1 + exp(-dot(biased_samples, weights) / g))


def train_slp(samples, answers, max_epochs, g, train_coeff, err_limit):
    num_n = len(answers[0])
    num_w = len(samples[0])

    weights = random.random((num_w + 1, num_n)) * 0.6 - 0.3
    biased_samples = hstack((samples, array([[1] for _ in samples])))
    
    err = inf
    epoch = 0
    while epoch < max_epochs and err > err_limit:
        y = 1 / (1 + exp(-dot(biased_samples, weights) / g))
        error = answers - y
        grad = error / g * y * (1 - y)

        wc = dot(biased_samples.transpose(), grad) * train_coeff
        weights += wc

        err = sum(error**2) / len(samples)

        epoch += 1
    
    return (epoch, err, weights)
