import numpy as np

from models.lcdm import *
from src.calculate_critical_densityContrast import crit_densityContrast_LCDM
from src.calculate_delta_inf import delta_inf
from src.calculate_initial_conditions import delta_inits_LCDM
from src.tools import sn


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                               # initial scale factor
a_col_vals = np.arange(0.01, 1.01, 0.01)   # collapse scale factor values

Om0_lcdm = Om0_finder_LCDM()               # LCDM matter density parameter

# ----------------------------------------
file = open(r'log/lmdp_LCDM_data.txt', 'x')
file.write('{0:15}\t\t{1:15}\t\t{2:15}\t\t{3:15}\t\t{4:15}\t\t{5:15}\n'.format('a_ini',
                                                                               'a_col',
                                                                               'delta_ini',
                                                                               'delta_prime_ini',
                                                                               'delta_c',
                                                                               'delta_inf'))
file.close()

for a_col_i in a_col_vals:

    # To increase the accuracy, and also in order to make the code stable, I have decided to divide
    # the error parts for each a_col interval. While I could have just set error to 1e-1 for all cases,
    # this way for smaller a_col values, the values becomes more accurate.
    if a_col_i < 0.15:
        delta_inf_err_i = 1e-4
    elif 0.15 <= a_col_i < 0.30:
        delta_inf_err_i = 1e-3
    elif 0.30 <= a_col_i < 0.60:
        delta_inf_err_i = 1e-2
    elif 0.60 <= a_col_i <= 1.0:
        delta_inf_err_i = 1e-1

    delta_inf_i = delta_inf(a_ini, a_col_i)
    delta_initials_lcdm = delta_inits_LCDM(a_ini, a_col_i, Om0_lcdm, delta_inf_i, delta_inf_err_i)
    delta_c_lcdm_i = crit_densityContrast_LCDM(a_ini, a_col_i, Om0_lcdm,
                                               delta_ini=delta_initials_lcdm[0],
                                               delta_prime_ini=delta_initials_lcdm[1])

    file = open(r'log/lmdp_LCDM_data.txt', 'a')
    file.write(
        '{0:7}\t\t{1:7}\t\t{2:7}\t\t{3:7}\t\t{4:7}\t\t{5:7}\n'.format(sn(a_ini,7),
                                                                      sn(a_col_i, 7),
                                                                      sn(delta_initials_lcdm[0], 7),
                                                                      sn(delta_initials_lcdm[1], 7),
                                                                      sn(delta_c_lcdm_i, 7),
                                                                      sn(delta_inf_i, 7)))

file.close()
