from math import exp
from numpy import *

def read_table(filename):
    with open(filename) as f:
        return array([[float(i) for i in line.split()] for line in f])


def run(samples, weights, g):
    outputs = []

    prev_out = hstack((samples, array([[1] for _ in samples])))
    for w in weights:
        out = 1 / (1 + exp(-dot(prev_out, w.transpose()) / g))
        outputs.append(out)
        prev_out = hstack((out, array([[1] for _ in out])))

    return outputs


def train_mlp(samples, answers, sizes, max_epochs, g, train_coeff, err_limit):
    weights = []

    assert sizes[0] == len(samples[0])
    assert sizes[-1] == len(answers[0])

    fixed_sizes = [len(samples[0])] + sizes
    for neur_prev, neur_curr in zip(fixed_sizes[:-1], fixed_sizes[1:]):
        layer = random.random((neur_curr, neur_prev + 1)) * 0.6 - 0.3
        weights.append(layer)

    # implementation of run() works, returns all layer outputs
    # TODO: implement backprop

    err = inf
    epoch = 0
    while epoch < max_epochs and err > err_limit:
        y = 1 / (1 + exp(-dot(biased_samples, weights.transpose()) / g))
        error = answers - y
        grad = error / g * y * (1 - y)

        wc = dot(transpose(grad), biased_samples) * train_coeff
        weights += wc

        err = sum(error**2) / len(samples)

        epoch += 1
    
    return (epoch, err, weights)
