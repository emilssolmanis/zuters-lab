import os
import sys
import lab7
from numpy import *

def test():
    lab_root = os.path.dirname(__file__)
    sys.path.append(lab_root + "../")
    lab4 = __import__("lab4.lab4")
    lab4 = lab4.lab4
    
    print "--------------------------------------------------------------------------------"
    print "----------------------------------LAB7------------------------------------------"
    print "--------------------------------------------------------------------------------"

    samples = lab4.read_table(lab_root + "/examples.txt")
    mins = apply_along_axis(min, 0, samples)
    maxes = apply_along_axis(max, 0, samples)
    norm_samples = (samples - mins) / (maxes - mins)
    answers = lab4.read_table(lab_root + "/d.txt")
    mins = apply_along_axis(min, 0, answers)
    maxes = apply_along_axis(max, 0, answers)
    norm_answers = (answers - mins) / (maxes - mins)

    slp_conf = {"max_epochs": 500, "g": 0.2, "train_coeff": 0.1, "err_limit": 0.01}
    dists = array([[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]])
    kohonen_conf = {"s_start": 1.0, "s_end": 0.2, "learn_start": 0.1, "learn_end": 0.01, "delta": 0.01, "num_clusters": 4, "distances": dists, "max_epochs": 1000}
    
    results = lab7.rbf(norm_samples, norm_answers, 0.5, kohonen_conf, slp_conf)
    results = results * (maxes - mins) + mins

    print "RBF trained, MSE: %.4f" % (sum((answers - results)**2) / len(answers))
