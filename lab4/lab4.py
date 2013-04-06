from numpy import *

def read_table(filename):
    with open(filename) as f:
        return array([[float(i) for i in line.split()] for line in f])

def kmeans(samples, num_clusters, max_epochs):
    num_rows, num_cols = samples.shape
    assert num_cols > 0 and num_rows > 0

    closest = zeros(num_rows)
    mins = apply_along_axis(min, 0, samples)
    maxes = apply_along_axis(max, 0, samples)
    cluster_centers = random.random((num_clusters, num_cols)) * (maxes - mins) + mins
    prev_cluster_centers = zeros((num_clusters, num_cols))
    
    epoch = 0
    while epoch < max_epochs and any(prev_cluster_centers != cluster_centers):
        prev_cluster_centers = cluster_centers
        for s_idx, sample in enumerate(samples):
            cluster = apply_along_axis(linalg.norm, 1, cluster_centers - sample).argmin()
            closest[s_idx] = cluster

        cluster_centers = zeros((num_clusters, num_cols))
        for i in range(len(cluster_centers)):
            cluster_samples = samples[where(closest == i)]
            if len(cluster_samples):
                cluster_centers[i] = apply_along_axis(sum, 0, cluster_samples) / len(cluster_samples)

        epoch += 1
    
    return (cluster_centers, epoch)

def kohonen(samples, s_low, s_high, learn_low, learn_high, delta, num_clusters, max_epochs):
    mins = apply_along_axis(min, 0, samples)
    maxes = apply_along_axis(max, 0, samples)
    pass
