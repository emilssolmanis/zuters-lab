import os
import sys
from numpy import *

def rbf(samples, answers, rbf_sigma, cluster_conf, slp_conf):
    lab_root = os.path.dirname(__file__)
    sys.path.append(lab_root + "../")
    lab2 = __import__("lab2.lab2")
    lab2 = lab2.lab2
    lab4 = __import__("lab4.lab4")
    lab4 = lab4.lab4
    
    num_rows, num_cols = samples.shape

    clusters = lab4.kohonen(samples, **cluster_conf)
    
    # creates a matrix with distance rows (sample1 to cluster1, sample1 to cluster2, ...)
    sample_dists = reshape(apply_along_axis(linalg.norm, 1, reshape(tile(samples, len(clusters)) - reshape(clusters, (1, -1)), (-1, num_cols))), (-1, len(clusters)))

    slp_inputs = exp(-sample_dists**2 / (2 * rbf_sigma**2))
    epoch, err, weights = lab2.train_slp(slp_inputs, answers, **slp_conf)
    
    return lab2.run(slp_inputs, weights, slp_conf['g'])
