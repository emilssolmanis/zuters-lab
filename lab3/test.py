import os
import numpy

def test():
    import lab3
    lab_root = os.path.dirname(__file__)
    numpy.set_printoptions(precision=3, suppress=True)

    print "--------------------------------------------------------------------------------"
    print "----------------------------------LAB3------------------------------------------"
    print "--------------------------------------------------------------------------------"

    g = 0.2
    samples = lab3.read_table(lab_root + "/examples_boolean.txt")
    answers = lab3.read_table(lab_root + "/d_notand.txt")
    epoch, err, weights = lab3.train_mlp(samples, answers, [2, 1], 500, g, 0.1, 0.01)
    print "Trained on d_notand.txt in %d epochs, MSE %.3f" % (epoch, err)
    results = lab3.run(samples, weights, g)
    for inp, expected, out in zip(samples, answers, results[-1][:, :-1]):
        print "%s : %s : %.3f" % (inp, expected, out)

    answers = lab3.read_table(lab_root + "/d_xor.txt")
    epoch, err, weights = lab3.train_mlp(samples, answers, [2, 1], 500, g, 0.1, 0.01)
    print "Trained on d_xor.txt in %d epochs, MSE %.3f" % (epoch, err)
    results = lab3.run(samples, weights, g)
    for inp, expected, out in zip(samples, answers, results[-1][:, :-1]):
        print "%s : %s : %.3f" % (inp, expected, out)

    samples = lab3.read_table(lab_root + "/examples.txt")
    answers = lab3.read_table(lab_root + "/d.txt")
    norm_samples = (samples - 1) / (samples.max() - 1)
    norm_answers = (answers - 1) / (answers.max() - 1)
    epoch, err, weights = lab3.train_mlp(norm_samples, norm_answers, [4, 1], 500, g, 0.1, 0.01)
    print "Trained on d.txt (0, 0.5, 1) in %d epochs, MSE %.3f" % (epoch, err)
    results = lab3.run(norm_samples, weights, g)
    for inp, expected, out in zip(norm_samples, norm_answers, results[-1][:, :-1]):
        print "%s : %s : %.3f" % (inp, expected, out)

    norm_samples = (samples == 3) * 0.9 + (samples == 2) * 0.5 + (samples == 1) * 0.1
    norm_answers = (answers == 3) * 0.9 + (answers == 2) * 0.5 + (answers == 1) * 0.1
    epoch, err, weights = lab3.train_mlp(norm_samples, norm_answers, [4, 1], 500, g, 0.1, 0.01)
    print "Trained on d.txt (0.1, 0.5, 0.9) in %d epochs, MSE %.3f" % (epoch, err)
    results = lab3.run(norm_samples, weights, g)
    for inp, expected, out in zip(norm_samples, norm_answers, results[-1][:, :-1]):
        print "%s : %s : %.3f" % (inp, expected, out)

    g = 0.3
    answers = lab3.read_table(lab_root + "/d2.txt")
    norm_samples = (samples - 1) / (samples.max() - 1)
    epoch, err, weights = lab3.train_mlp(norm_samples, answers, [4, 3], 500, g, 0.01, 0.01)
    print "Trained on d2.txt (0, 1) in %d epochs, MSE %.3f" % (epoch, err)
    results = lab3.run(norm_samples, weights, g)
    for inp, expected, out in zip(norm_samples, answers, results[-1][:, :-1]):
        print "%s : %s : %s" % (inp, expected, out)

    norm_samples = (samples == 3) * 0.9 + (samples == 2) * 0.5 + (samples == 1) * 0.1
    norm_answers = (answers == 1) * 0.9 + (answers == 0) * 0.1
    epoch, err, weights = lab3.train_mlp(norm_samples, norm_answers, [4, 3], 500, g, 0.1, 0.01)
    print "Trained on d2.txt (0.1, 0.9) in %d epochs, MSE %.3f" % (epoch, err)
    results = lab3.run(norm_samples, weights, g)
    for inp, expected, out in zip(norm_samples, norm_answers, results[-1][:, :-1]):
        print "%s : %s : %s" % (inp, expected, out)
    
