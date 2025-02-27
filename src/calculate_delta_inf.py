import numpy as np
import scipy as sp


def delta_inf(a_ini, a_col):
    C_eds = (3 / 5) * (3 * np.pi / 2) ** (2 / 3) * a_col ** (-1)
    delta_ini_eds = C_eds * a_ini
    delta_prime_ini_eds = C_eds

    def nonLin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        E = np.sqrt(a ** (-3))
        E_prime = (-3 * a ** (-4)) / (2 * E)

        Om = (a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 4 / 3
        c3 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * (delta_prime ** 2 / (1 + delta)) + c3 * delta * (1 + delta)]

    res = sp.integrate.solve_ivp(nonLin_mdp, (a_ini, a_col), [delta_ini_eds, delta_prime_ini_eds],
                                method='Radau', t_eval=[a_col], atol=1e-8, rtol=1e-8)

    if res['success']:
        delta_infinity = res['y'][0, 0]
        return delta_infinity
