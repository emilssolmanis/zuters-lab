import os
import sys
import random
import numpy as np

def test():
    lab_root = os.path.dirname(__file__)
    sys.path.append(lab_root + "../")
    lab3 = __import__("lab3.lab3")
    lab3 = lab3.lab3
    
    print "--------------------------------------------------------------------------------"
    print "----------------------------------LAB5------------------------------------------"
    print "--------------------------------------------------------------------------------"

    samples = lab3.read_table(lab_root + "/examples.txt")
    answers = lab3.read_table(lab_root + "/d.txt")

    assert len(samples) == len(answers)
    
    cums = map(round, np.hstack((np.array([0]), np.cumsum([len(samples) / 5.0 for _ in range(5)]))))
    sizes = map(lambda x, y: int(x - y), cums[1:], cums[:-1])

    indice_pool = set(range(len(samples)))
    
    sample_indices = []
    for size in sizes:
        indices = np.array(sorted(random.sample(indice_pool, size)))
        map(indice_pool.remove, indices)
        sample_indices.append(indices)
    
    max_epochs = 5000
    err_limit = 0.001
    gs = [0.05, 0.1, 0.2, 0.3, 0.5]
    etas = [0.05, 0.1, 0.2]
    sizes = [[2, 2, 1], [2, 3, 1], [2, 4, 1]]
    
    def get_confs():
        for i in range(5):
            for g in gs:
                for eta in etas:
                    for size in sizes:
                        yield (i, g, eta, size)

    def run_conf(conf):
        i, g, eta, size = conf
        train_indices = reduce(lambda x, y: np.concatenate((x, y)), sample_indices[:i] + sample_indices[i+1:])
        validate_indices = sample_indices[i]
        epoch, err, weights = lab3.train_mlp(samples[train_indices], answers[train_indices], size, max_epochs, g, eta, err_limit)
        results = lab3.run(samples[validate_indices], weights, g)
        if epoch < max_epochs:
            err = sum((answers[validate_indices] - results[-1][:, :-1])**2) / len(validate_indices)
            return err
        else:
            return float('Inf')

    print "Val. grp.\tSlope\tLearn coeff.\tTopology\tMSE"
    for conf in get_confs():
        i, g, eta, size = conf
        print "%d\t%.3f\t%.3f\t%s\t%.4f" % (i, g, eta, size, run_conf(conf))

    # TODO: create a table, actually find the best confs
    # TODO: parallelize this crap
    # results = map(run_conf, get_confs())
