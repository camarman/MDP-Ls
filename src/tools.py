import numpy as np


def sn(num, significant_figures):
    return f"{num:.{significant_figures - 1}e}"


def diracDeltaApprox(a, var_eps=1e-4):
    return (1 / np.pi) * (var_eps / (a ** 2 + var_eps ** 2))
