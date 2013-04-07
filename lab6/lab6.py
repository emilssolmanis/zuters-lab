from numpy import *

def __most_common_class(samples):
    return int(max({len(filter(lambda x: x[0] == c, samples)): c for c in set([s[0] for s in samples])}.iteritems())[1])


def read_table(filename):
    with open(filename) as f:
        return array([[float(i) for i in line.split()] for line in f])


class Complex(object):
    def __init__(self, attr=None, val=None):
        self.rules = []
        if attr is not None and val is not None:
            self.rules.append((attr, val))

    def __nonzero__(self):
        return self.rules

    def __and__(self, other):
        res = Complex()
        res.rules = self.rules + other.rules
        return res

    def __repr__(self):
        return " AND ".join("%d == %d" % (rule[0], rule[1]) for rule in self.rules)

    def __eq__(self, other):
        return self.rules == other.rules

    def __len__(self):
        return len(self.rules)

    def format(self, attr_names):
        return " AND ".join("%s == %s" % (attr_names[rule[0]][0], attr_names[rule[0]][rule[1]]) for rule in self.rules)

    def consistent(self):
        return len(set(rule[0] for rule in self.rules)) == len(self.rules)


def split_covered(cpx, samples):
    try:
        covered = array(reduce(intersect1d, (where(samples[:, attr] == val)[0] for attr, val in cpx.rules)))
    except TypeError:
        covered = array([], dtype=int)
    uncovered = setdiff1d(array(range(len(samples))), covered)
    return (samples[covered], samples[uncovered])


def statistical_significance(cpx, samples, classes):
    covered, _ = split_covered(cpx, samples)
    covered_hist = array([size(where(covered[:, 0] == c)) for c in classes])
    overall_hist = array([size(where(samples[:, 0] == c)) for c in classes], dtype=float64)
    overall_hist *= float(len(covered)) / len(samples)
    return 2 * sum(f * log2(f / g) for f, g in zip(covered_hist, overall_hist) if f != 0 and g != 0)


def laplace(cpx, samples, classes):
    covered, _ = split_covered(cpx, samples)
    if not size(covered):
        return 0.0
    c = __most_common_class(covered)
    n_c = size(where(covered[:, 0] == c))
    n_tot = len(covered)
    return float(n_c + 1) / (n_tot + len(classes))


def find_best_cpx(samples, max_star, classes, stat_sig_thr):
    star = [Complex()]
    sels = [Complex(attr_idx + 1, int(val)) for attr_idx, col in enumerate(samples[:, 1:].transpose()) for val in unique(col)]

    best_cpx = Complex()
    while star:
        star2 = [st & sel for st in star for sel in sels]
        star2 = filter(lambda x: x not in star and x.consistent(), star2)
        for cpx in star2:
            if statistical_significance(cpx, samples, classes) >= stat_sig_thr:
                if laplace(cpx, samples, classes) >= laplace(best_cpx, samples, classes):
                    best_cpx = cpx
        star = sorted(star2, key=lambda cpx: laplace(cpx, samples, classes), reverse=True)[:max_star]

    return best_cpx


def cn2(samples, attr_names, max_star, stat_sig_thr=0.0):
    classes = unique(samples[:, 0])
    rule_list = []
    c = __most_common_class(samples)
    default_rule = 'ELSE <%s = %s>' % (attr_names[0][0], attr_names[0][c])

    while True:
        best_cpx = find_best_cpx(samples, max_star, classes, stat_sig_thr)
        if len(best_cpx) > 0:
            covered_samples, uncovered_samples = split_covered(best_cpx, samples)
            c = __most_common_class(covered_samples)
            rule = 'IF <%s> \n\tTHEN <%s = %s>' % (best_cpx.format(attr_names), attr_names[0][0], attr_names[0][c])
            rule_list.append(rule)
            samples = uncovered_samples
        if not len(best_cpx) or not len(samples):
            break

    rule_list.append(default_rule)
    return rule_list
