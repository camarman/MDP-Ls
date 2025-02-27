import numpy as np
import scipy as sp

from src.calculate_delta_inf import delta_inf
from src.tools import diracDeltaApprox


# step size
h = 1e-5


def delta_nonLin_LsCDM(a_ini, a_dag, a_end, R_lscdm, delta_ini, delta_prime_ini):

    def ddelta_prime_da(a, delta, delta1, R_lscdm, a_dag):
        dirac_delta = diracDeltaApprox(a - a_dag)
        term1 = - (3 / (2 * a)) * (2 - (a ** -3 - (2 / 3) * a * R_lscdm * dirac_delta) / (
                a ** -3 + R_lscdm * np.sign(a - a_dag))) * delta1
        term2 = (4 / 3) * (delta1 ** 2 / (1 + delta))
        term3 = (3 / (2 * a ** 2)) * (a ** -3 / (a ** -3 + R_lscdm * np.sign(a - a_dag))) * delta * (1 + delta)
        return term1 + term2 + term3

    def ddelta_da(a, delta, delta1):
        return delta1

    a_values = [a_ini]
    delta_values = [delta_ini]
    delta_prime_values = [delta_prime_ini]

    while a_values[-1] <= a_end:   # runs the code until a_end
        a = a_values[-1]
        delta = delta_values[-1]
        delta1 = delta_prime_values[-1]

        k1_delta = h * ddelta_da(a, delta, delta1)
        k1_delta1 = h * ddelta_prime_da(a, delta, delta1, R_lscdm, a_dag)

        k2_delta = h * ddelta_da(a + 0.5 * h, delta + 0.5 * k1_delta, delta1 + 0.5 * k1_delta1)
        k2_delta1 = h * ddelta_prime_da(a + 0.5 * h, delta + 0.5 * k1_delta, delta1 + 0.5 * k1_delta1, R_lscdm, a_dag)

        k3_delta = h * ddelta_da(a + 0.5 * h, delta + 0.5 * k2_delta, delta1 + 0.5 * k2_delta1)
        k3_delta1 = h * ddelta_prime_da(a + 0.5 * h, delta + 0.5 * k2_delta, delta1 + 0.5 * k2_delta1, R_lscdm, a_dag)

        k4_delta = h * ddelta_da(a + h, delta + k3_delta, delta1 + k3_delta1)
        k4_delta1 = h * ddelta_prime_da(a + h, delta + k3_delta, delta1 + k3_delta1, R_lscdm, a_dag)

        a_new = a + h
        delta_new = delta + (k1_delta + 2 * k2_delta + 2 * k3_delta + k4_delta) / 6
        delta1_new = delta1 + (k1_delta1 + 2 * k2_delta1 + 2 * k3_delta1 + k4_delta1) / 6

        a_values.append(a_new)
        delta_values.append(delta_new)
        delta_prime_values.append(delta1_new)

    return a_values, delta_values


def col_params_LsCDM(a_ini, a_dag, a_end, R_lscdm, delta_ini, delta_prime_ini):
    a_values_lscdm, delta_values_lscdm = delta_nonLin_LsCDM(a_ini, a_dag, a_end, R_lscdm, delta_ini, delta_prime_ini)
    for i in range(len(delta_values_lscdm)-1, 0, -1):
        a_col_i = a_values_lscdm[i]
        delta_inf_i = delta_inf(a_ini, a_col_i)
        if delta_values_lscdm[i] <= delta_inf_i:
            a_col_true = a_values_lscdm[i]
            delta_num_nonLin = delta_values_lscdm[i]
            return a_col_true, delta_num_nonLin
