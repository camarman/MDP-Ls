import numpy as np
import scipy as sp

from src.tools import diracDeltaApprox


def crit_densityContrast_EdS(a_ini, a_col, delta_ini, delta_prime_ini):

    def lin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        E = np.sqrt(a ** (-3))
        E_prime = (-3 * a ** (-4)) / (2 * E)

        Om = (a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * delta]

    res = sp.integrate.solve_ivp(lin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                method='Radau', t_eval=[a_col], atol=1e-8, rtol=1e-8)
    delta_c = res['y'][0, 0]
    return delta_c


def crit_densityContrast_LCDM(a_ini, a_col, Om0, delta_ini, delta_prime_ini):

    def lin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0))
        E_prime = (-3 * Om0 * a ** (-4)) / (2 * E)

        Om = (Om0 * a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * delta]

    res = sp.integrate.solve_ivp(lin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                method='Radau', t_eval=[a_col], atol=1e-8, rtol=1e-8)
    delta_c = res['y'][0, 0]
    return delta_c


def crit_densityContrast_LsCDM(a_ini, a_col, a_dag, Om0, delta_ini, delta_prime_ini):

    def lin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        dirac_delta = diracDeltaApprox(a - a_dag)
        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0) * np.sign(a - a_dag))
        E_prime = (-3 * Om0 * a ** (-4) + (1 - Om0) * 2 * dirac_delta) / (2 * E)

        Om = (Om0 * a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * delta]

    res = sp.integrate.solve_ivp(lin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                method='Radau', t_eval=[a_col], atol=1e-8, rtol=1e-8)
    delta_c = res['y'][0, 0]
    return delta_c