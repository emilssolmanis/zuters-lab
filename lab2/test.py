import os

def test():
    print "--------------------------------------------------------------------------------"

    import lab2
    lab2_root = os.path.dirname(__file__)

    g = 0.2
    samples = lab2.read_table(lab2_root + "/examples.txt")

    answers = lab2.read_table(lab2_root + "/d.txt")
    epoch, err, weights = lab2.train_slp(samples, answers, 500, g, 0.1, 0.01)
    print "Trained on d.txt in %d epochs, MSE %.3f" % (epoch, err)
    results = lab2.run(samples, weights, g)
    for inp, expected, out in zip(samples, answers, results):
        print "%s : %s : %.3f" % (inp, expected, out)

    answers = lab2.read_table(lab2_root + "/d2.txt")
    epoch, err, weights = lab2.train_slp(samples, answers, 500, g, 0.1, 0.01)
    print "Trained on d2.txt in %d epochs, MSE %.3f" % (epoch, err)
    results = lab2.run(samples, weights, g)
    for inp, expected, out in zip(samples, answers, results):
        print "%s : %s : %s" % (inp, expected, ["%.3f" % f for f in out])

    answers = lab2.read_table(lab2_root + "/d3.txt")
    epoch, err, weights = lab2.train_slp(samples, answers, 500, g, 0.1, 0.01)
    print "Trained on d3.txt in %d epochs, MSE %.3f" % (epoch, err)
    results = lab2.run(samples, weights, g)
    for inp, expected, out in zip(samples, answers, results):
        print "%s : %s : %s" % (inp, expected, ["%.3f" % f for f in out])

    answers = lab2.read_table(lab2_root + "/d.txt")

    g = 0.1
    epoch, err, weights = lab2.train_slp(samples, answers, 500, g, 0.1, 0.01)
    print "Trained on d.txt, g = 0.1 in %d epochs, MSE %.3f" % (epoch, err)
    results = lab2.run(samples, weights, g)
    for inp, expected, out in zip(samples, answers, results):
        print "%s : %s : %.3f" % (inp, expected, out)

    g = 0.3
    epoch, err, weights = lab2.train_slp(samples, answers, 500, g, 0.1, 0.01)
    print "Trained on d.txt, g = 0.3 in %d epochs, MSE %.3f" % (epoch, err)
    results = lab2.run(samples, weights, g)
    for inp, expected, out in zip(samples, answers, results):
        print "%s : %s : %.3f" % (inp, expected, out)

    g = 0.03
    epoch, err, weights = lab2.train_slp(samples, answers, 500, g, 0.1, 0.01)    
    prev_epoch = epoch
    while epoch <= prev_epoch:
        prev_epoch = epoch
        epoch, err, weights = lab2.train_slp(samples, answers, 500, g, 0.1, 0.01)
        g += 0.001
    
    print "Minimum g = %.3f, %d epochs" % (g, prev_epoch)
