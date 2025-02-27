import numpy as np
import scipy as sp

from src.tools import diracDeltaApprox


def delta_inits_LCDM(a_ini, a_col, Om0, delta_inf, delta_inf_err):

    def nonLin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0))
        E_prime = (-3 * Om0 * a ** (-4)) / (2 * E)

        Om = (Om0 * a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 4 / 3
        c3 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * (delta_prime ** 2 / (1 + delta)) + c3 * delta * (1 + delta)]

    flag = True

    log_delta_ini_max = 0.0
    log_delta_ini_min = -12.0

    while flag:
        log_delta_ini = (log_delta_ini_max + log_delta_ini_min) / 2
        delta_ini = np.exp(log_delta_ini)  # C * a_ini
        delta_prime_ini = delta_ini / a_ini  # C
        res = sp.integrate.solve_ivp(nonLin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                    method='Radau', t_eval=[a_col], atol=1e-8, rtol=1e-8)
        if res['success']:
            deltaMatter_nonLin = res['y'][0, 0]
            if abs(deltaMatter_nonLin - delta_inf) > delta_inf_err:
                if deltaMatter_nonLin > delta_inf:
                    log_delta_ini_max = log_delta_ini
                else:
                    log_delta_ini_min = log_delta_ini
            else:
                break
        else:
            log_delta_ini_max = log_delta_ini

    return delta_ini, delta_prime_ini


def delta_inits_LsCDM(a_ini, a_col, a_dag, Om0, delta_inf, delta_inf_err):

    def nonLin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        dirac_delta = diracDeltaApprox(a - a_dag)
        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0) * np.sign(a - a_dag))
        E_prime = (-3 * Om0 * a ** (-4) + (1 - Om0) * 2 * dirac_delta) / (2 * E)

        Om = (Om0 * a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 4 / 3
        c3 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * (delta_prime ** 2 / (1 + delta)) + c3 * delta * (1 + delta)]

    flag = True

    log_delta_ini_max = 0.0
    log_delta_ini_min = -12.0

    while flag:
        log_delta_ini = (log_delta_ini_max + log_delta_ini_min) / 2
        delta_ini = np.exp(log_delta_ini)  # C * a_ini
        delta_prime_ini = delta_ini / a_ini  # C
        res = sp.integrate.solve_ivp(nonLin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                    method='Radau', t_eval=[a_col], atol=1e-8, rtol=1e-8)
        if res['success']:
            deltaMatter_nonLin = res['y'][0, 0]
            if abs(deltaMatter_nonLin - delta_inf) > delta_inf_err:
                if deltaMatter_nonLin > delta_inf:
                    log_delta_ini_max = log_delta_ini
                else:
                    log_delta_ini_min = log_delta_ini
            else:
                break
        else:
            log_delta_ini_max = log_delta_ini

    return delta_ini, delta_prime_ini
