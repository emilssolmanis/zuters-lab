from math import exp
from numpy import *

def read_table(filename):
    with open(filename) as f:
        return array([[float(i) for i in line.split()] for line in f])


def run(samples, weights, g):
    outputs = []

    prev_out = hstack((samples, array([[1] for _ in samples])))
    outputs.append(prev_out)
    for w in weights:
        out = 1 / (1 + exp(-dot(prev_out, w) / g))
        prev_out = hstack((out, array([[1] for _ in out])))
        outputs.append(prev_out)

    return outputs


def train_mlp(samples, answers, sizes, max_epochs, g, train_coeff, err_limit):
    weights = []

    assert sizes[0] == len(samples[0])
    assert sizes[-1] == len(answers[0])

    fixed_sizes = [len(samples[0])] + sizes
    for neur_prev, neur_curr in zip(fixed_sizes[:-1], fixed_sizes[1:]):
        layer = random.random((neur_prev + 1, neur_curr)) * 0.6 - 0.3
        weights.append(layer)

    # TODO: check the whole part about hstacking the answer, seems to act weird
    err = inf
    epoch = 0
    while epoch < max_epochs and err > err_limit:
        y = run(samples, weights, g)

        output_error = answers - y[-1][:, :-1]
        output_error = hstack((output_error, zeros((len(output_error), 1))))
        get_error = lambda: output_error
        for i in range(len(weights) - 1, -1, -1):
            error = get_error()
            grad = error / g * y[i + 1] * (1 - y[i + 1])
            wc = dot(y[i].transpose(), grad) * train_coeff
            wc = wc[:, :-1]
            weights[i] += wc
            get_error = lambda: dot(grad[:, :-1], weights[i + 1].transpose())

        err = sum(output_error[:, :-1]**2) / size(output_error[:, :-1])

        epoch += 1
    
    return (epoch, err, weights)
