import os
import lab6

def test():
    print "--------------------------------------------------------------------------------"
    print "----------------------------------LAB6------------------------------------------"
    print "--------------------------------------------------------------------------------"

    lab_root = os.path.dirname(__file__)
    samples = lab6.read_table(lab_root + "/examples.txt");    

    with open(lab_root + "/attributevaluenames.txt") as f:
        attr_names = [line.split() for line in f]

    print "CN2, statistical significance 0.0"
    for rule in lab6.cn2(samples, attr_names, 5, 0.0):
        print rule
    
    print "CN2, statistical significance 1.0"
    for rule in lab6.cn2(samples, attr_names, 5, 1.0):
        print rule
