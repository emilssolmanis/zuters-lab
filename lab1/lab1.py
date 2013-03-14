# risk, hist, dept, recm, income
from math import log

def __most_common_class(samples):
    return max({len(filter(lambda x: x[0] == c, samples)): c for c in set([s[0] for s in samples])}.iteritems())[1]

class Node(object):
    def __init__(self):
        self.value = None
        self.attribute = None
        self.node_class = None
        self.nodes = []


def read_table(filename):
    with open(filename) as f:
        return [[int(i) for i in line.split()] for line in f]


def entropy(samples):
    ent = 0
    for c in set(samples):
        c_count = len(filter(lambda x: x == c, samples))
        pc = float(c_count) / len(samples)
        ent -= pc * log(pc, 2)
    return ent


def best_attribute(samples, attributes):
    best = None
    for a in attributes:
        entropy_a = 0
        for value in set(sample[a] for sample in samples):
            e_a_v = filter(lambda x: x[a] == value, samples)
            entropy_a += len(e_a_v) * entropy([s[0] for s in e_a_v])
        if best is None or entropy_a < entropy_best:
            best = a
            entropy_best = entropy_a
    return best


def id3(samples, attributes):
    t = Node()
    if len(set(s[0] for s in samples)) == 1:
        t.node_class = samples[0][0]
    elif not attributes:
        t.node_class = __most_common_class(samples)
    else:
        best = best_attribute(samples, attributes)
        t.attribute = best
        for value in set(s[best] for s in samples):
            n = Node()
            e_b_v = filter(lambda x: x[best] == value, samples)
            if not e_b_v:
                n.node_class = __most_common_class(samples)
            else:
                n = id3(e_b_v, attributes - {best})
            n.value = value
            t.nodes.append(n)
    return t


def classify(tree, sample):
    if tree.node_class:
        return tree.node_class
    else:
        a = tree.attribute
        v = sample[a]
        subt = next(t for t in tree.nodes if t.value == v)
        return classify(subt, sample)
