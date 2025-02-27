import numpy as np

from src.calculate_critical_densityContrast import crit_densityContrast_EdS
from src.calculate_delta_inf import delta_inf
from src.tools import sn


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                               # initial scale factor
a_col_vals = np.arange(0.01, 1.01, 0.01)   # collapse scale factor values

# ----------------------------------------
file = open(r'log/lmdp_EdS_data.txt', 'x')
file.write('{0:15}\t\t{1:15}\t\t{2:15}\t\t{3:15}\t\t{4:15}\t\t{5:15}\n'.format('a_ini',
                                                                               'a_col',
                                                                               'delta_ini',
                                                                               'delta_prime_ini',
                                                                               'delta_c',
                                                                               'delta_inf'))
file.close()

for a_col_i in a_col_vals:
    C_eds = (3 / 5) * (3 * np.pi / 2) ** (2 / 3) * a_col_i ** (-1)
    delta_ini_eds = C_eds * a_ini
    delta_prime_ini_eds = C_eds

    delta_inf_i = delta_inf(a_ini, a_col_i)
    delta_c_EdS_i = crit_densityContrast_EdS(a_ini, a_col_i,
                                          delta_ini=delta_ini_eds,
                                          delta_prime_ini=delta_prime_ini_eds)

    file = open(r'log/lmdp_EdS_data.txt', 'a')
    file.write('{0:7}\t\t{1:7}\t\t{2:7}\t\t{3:7}\t\t{4:7}\t\t{5:7}\n'.format(sn(a_ini,7),
                                                                             sn(a_col_i, 7),
                                                                             sn(delta_ini_eds, 7),
                                                                             sn(delta_prime_ini_eds, 7),
                                                                             sn(delta_c_EdS_i, 7),
                                                                             sn(delta_inf_i, 7)))

file.close()
