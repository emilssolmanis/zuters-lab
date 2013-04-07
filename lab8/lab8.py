import sys
from numpy import *

def read_table(filename):
    with open(filename) as f:
        return array([[float(i) for i in line.split()] for line in f])


def feature_selection(samples, answers, sizes, max_epochs, g, train_coeff, err_limit):
    sys.path.append("../")
    lab3 = __import__("lab3.lab3")
    lab3 = lab3.lab3

    mins = apply_along_axis(min, 0, samples)
    maxes = apply_along_axis(max, 0, samples)
    norm_samples = (samples - mins) / (maxes - mins)
    
    mins = apply_along_axis(min, 0, answers)
    maxes = apply_along_axis(max, 0, answers)
    norm_answers = (answers - mins) / (maxes - mins)

    num_samples, num_attrs = samples.shape
    errors = []
    epochs = []

    epoch, err, weights = lab3.train_mlp(norm_samples, norm_answers, [sizes[0] + 1] + sizes[1:], max_epochs, g, train_coeff, err_limit)
    results = lab3.run(norm_samples, weights, g)
    errors.append(linalg.norm(results[-1][:, :-1] - norm_answers))
    epochs.append(epoch)
    
    for k in range(num_attrs):
        without_feature = hstack((norm_samples[:, :k], norm_samples[:, k+1:]))
        epoch, err, weights = lab3.train_mlp(without_feature, norm_answers, sizes, max_epochs, g, train_coeff, err_limit)
        results = lab3.run(without_feature, weights, g)
        errors.append(linalg.norm(results[-1][:, :-1] - norm_answers))
        epochs.append(epoch)

    return (array(epochs), array(errors))


def feature_extraction(samples, answers, var, max_components, single_comp_thr, max_epochs, g, train_coeff, err_limit):
    sys.path.append("../")
    lab3 = __import__("lab3.lab3")
    lab3 = lab3.lab3

    mins = apply_along_axis(min, 0, samples)
    maxes = apply_along_axis(max, 0, samples)
    norm_samples = ((samples - mins) / (maxes - mins)) * 2 - 1
    
    mins = apply_along_axis(min, 0, answers)
    maxes = apply_along_axis(max, 0, answers)
    norm_answers = (answers - mins) / (maxes - mins)

    num_samples, num_attrs = samples.shape
    
    var_thr = (1 - var) * num_attrs * (num_samples - 1)
    comp = 1
    
    variances = []

    working_samples = norm_samples
    scores = []
    loadings = []
    while comp <= max_components and linalg.norm(working_samples) >= var_thr:
        variances.append(linalg.norm(working_samples))
        t = working_samples[:, 0]
        while True:
            p = dot(norm_samples.transpose(), t)
            p /= linalg.norm(p)
            t_old = t
            t = dot(norm_samples, p)
            if linalg.norm(t - t_old) < single_comp_thr:
                break

        norm_samples -= dot(mat(t).transpose(), mat(p))

        if scores == []:
            scores = mat(t).transpose()
        else:
            scores = hstack((scores, mat(t).transpose()))

        if loadings == []:
            loadings = mat(p).transpose()
        else:
            loadings = hstack((loadings, mat(p).transpose()))

        comp += 1
    variances.append(linalg.norm(working_samples))
    
    scores = array(scores)
    loadings = array(loadings)
    epoch, err, weights = lab3.train_mlp(scores, norm_answers, [scores.shape[1], answers.shape[1]], max_epochs, g, train_coeff, err_limit)
    results = lab3.run(scores, weights, g)
    return (linalg.norm(results[-1][:, :-1] - norm_answers), array(variances), scores.shape[1])
