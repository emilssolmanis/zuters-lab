import os
from lab4 import read_table, kmeans, kohonen
from numpy import array

def test():
    print "--------------------------------------------------------------------------------"
    print "----------------------------------LAB4------------------------------------------"
    print "--------------------------------------------------------------------------------"

    lab_root = os.path.dirname(__file__)
    samples = read_table(lab_root + "/examples.txt");

    centers, epoch = kmeans(samples, 4, 1000)
    print "K-means finished in %d epochs"
    print "Found cluster centers: "
    for cluster in centers:
        print "\t%s" % cluster
    
    dists = array([[0, 1, 1, 2], [1, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]])
    centers = kohonen(samples, 1.0, 0.2, 0.1, 0.01, 0.01, 4, dists, 100)

    print "Found Kohonen SOM cluster centers: "
    for cluster in centers:
        print "\t%s" % cluster
