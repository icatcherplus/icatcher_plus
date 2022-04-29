import numpy as np
from scipy import stats


def bootstrap(x, confidence=0.95):
    # sample = np.random.choice(x, size=30, replace=False)
    bs = np.random.choice(x, (len(x), 1000), replace=True)
    bs_means = bs.mean(axis=0)
    bs_means_mean = bs_means.mean()
    minquant = (1 - confidence) / 2
    maxquant = minquant + confidence
    lower_ci = np.quantile(bs_means, minquant)
    upper_ci = np.quantile(bs_means, maxquant)
    return bs_means_mean, lower_ci, upper_ci


def t_test(x, y):
    a = np.array(x)
    b = np.array(y)
    t, p = stats.ttest_ind(a, b, equal_var=False, permutations=1000)

    # Compute the descriptive statistics of a and b.
    avar = a.var(ddof=1)
    na = a.size
    adof = na - 1

    bvar = b.var(ddof=1)
    nb = b.size
    bdof = nb - 1

    # Compute Welch's test dof (which scipy uses behind the scenes with equal_var=False)
    dof = (avar / na + bvar / nb) ** 2 / (avar ** 2 / (na ** 2 * adof) + bvar ** 2 / (nb ** 2 * bdof))
    return t, p, dof


def t_test_paired(x, y):
    t, p = stats.ttest_rel(x, y)
    dof = len(x) - 1
    return t, p, dof