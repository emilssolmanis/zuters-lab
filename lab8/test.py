import os
import lab8

def test():
    print "--------------------------------------------------------------------------------"
    print "----------------------------------LAB8------------------------------------------"
    print "--------------------------------------------------------------------------------"

    lab_root = os.path.dirname(__file__)
    samples = lab8.read_table(lab_root + "/examples.txt")
    answers = lab8.read_table(lab_root + "/d.txt")
    
    epochs, errors = lab8.feature_selection(samples, answers, [3, 1], 1000, 0.2, 0.05, 0.005)
    for epoch, err, excluded in zip(epochs, errors, range(len(errors))):
        print "Trained in %4d epochs, MSE %.4f without feature %d" % (epoch, err, excluded)
    
    error, variances, components = lab8.feature_extraction(samples, answers, 0.95, 4, 0.00001, 1000, 0.2, 0.05, 0.005)
    print "Extracted %d components, error %.4f" % (components, error)
    error, variances, components = lab8.feature_extraction(samples, answers, 0.95, 2, 0.00001, 1000, 0.2, 0.05, 0.005)
    print "Extracted %d components, error %.4f" % (components, error)
