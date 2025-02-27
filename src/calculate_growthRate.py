import numpy as np
import scipy as sp

from src.tools import diracDeltaApprox


def growthRate_EdS(a_eval, a_ini, a_col, f_ini):

    def df_da(a, f):
        E = np.sqrt(a ** (-3))
        E_prime = (-3 * a ** (-4)) / (2 * E)

        term1 = -f**2 / a
        term2 = -(2 / a + E_prime / E) * f
        term3 = 1.5 / (a**4 * E**2)
        return term1 + term2 + term3

    def f_eds(a_eval):
        res = sp.integrate.solve_ivp(df_da, (a_ini, a_col), [f_ini],
                                     method='Radau', t_eval=[a_eval], atol=1e-8, rtol=1e-8)
        return res['y'][0, 0]

    return f_eds(a_eval)


def growthRate_LCDM(a_eval, a_ini, a_col, Om0, f_ini):

    def df_da(a, f):
        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0))
        E_prime = (-3 * Om0 * a ** (-4)) / (2 * E)

        term1 = -f**2 / a
        term2 = -(2 / a + E_prime / E) * f
        term3 = 1.5 * Om0 / (a**4 * E**2)
        return term1 + term2 + term3

    def f_lcdm(a_eval):
        res = sp.integrate.solve_ivp(df_da, (a_ini, a_col), [f_ini],
                                     method='Radau', t_eval=[a_eval], atol=1e-8, rtol=1e-8)
        return res['y'][0, 0]

    return f_lcdm(a_eval)


def growthRate_LsCDM(a_eval, a_ini, a_col, a_dag, Om0, f_ini):

    def df_da(a, f):
        dirac_delta = diracDeltaApprox(a - a_dag)
        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0) * np.sign(a - a_dag))
        E_prime = (-3 * Om0 * a ** (-4) + 2 * (1 - Om0) * dirac_delta) / (2 * E)

        term1 = -f**2 / a
        term2 = -(2 / a + E_prime / E) * f
        term3 = 1.5 * Om0 / (a**4 * E**2)
        return term1 + term2 + term3

    def f_lscdm(a_eval):
        res = sp.integrate.solve_ivp(df_da, (a_ini, a_col), [f_ini],
                                     method='Radau', t_eval=[a_eval], atol=1e-8, rtol=1e-8)
        return res['y'][0, 0]

    return f_lscdm(a_eval)