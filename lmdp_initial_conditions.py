from models.lcdm import *
from models.lscdm import *

from src.calculate_critical_densityContrast import *
from src.calculate_delta_inf import delta_inf
from src.calculate_initial_conditions import delta_inits_LCDM, delta_inits_LsCDM
from src.tools import sn


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                          # initial scale factor
a_col = 1.0                         # collapse scale factor

Om0_lcdm = Om0_finder_LCDM()          # LCDM matter density parameter

z_dag = 1.7                           # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)               # LsCDM transition scale factor
Om0_lscdm = Om0_finder_LsCDM(z_dag)   # LsCDM matter density parameter


# ---------- initial conditions for an EdS Universe
C_eds = (3 / 5) * (3 * np.pi / 2) ** (2 / 3) * a_col ** (-1)
delta_ini_eds = C_eds * a_ini
delta_prime_ini_eds = C_eds


# To increase the accuracy, and also in order to make the code stable, I have decided to divide
# the error parts for each a_col interval. While I could have just set error to 1e-1 for all cases,
# this way for smaller a_col values, the values becomes more accurate.
if a_col < 0.15:
    numerical_inf_err = 1e-4
elif 0.15 <= a_col < 0.30:
    numerical_inf_err = 1e-3
elif 0.30 <= a_col < 0.60:
    numerical_inf_err = 1e-2
elif 0.60 <= a_col <= 1.0:
    numerical_inf_err = 1e-1


#####################################################
# -------------------- ANALYSIS --------------------
#####################################################
num_nonLin_dc = delta_inf(a_ini, a_col)
delta_initials_lcdm = delta_inits_LCDM(a_ini, a_col, Om0_lcdm,
                                       num_nonLin_dc,
                                       numerical_inf_err)

delta_initials_lscdm = delta_inits_LsCDM(a_ini, a_col, a_dag, Om0_lscdm,
                                         num_nonLin_dc,
                                         numerical_inf_err)

delta_c_EdS = crit_densityContrast_EdS(a_ini, a_col,
                                        delta_ini=delta_ini_eds,
                                        delta_prime_ini=delta_prime_ini_eds)

delta_c_LCDM = crit_densityContrast_LCDM(a_ini, a_col, Om0_lcdm,
                                        delta_ini=delta_initials_lcdm[0],
                                        delta_prime_ini=delta_initials_lcdm[1])

delta_c_LsCDM = crit_densityContrast_LsCDM(a_ini, a_col, a_dag, Om0_lscdm,
                                            delta_ini=delta_initials_lscdm[0],
                                            delta_prime_ini=delta_initials_lscdm[1])


# ####################################################
# # -------------------- PRINTING --------------------
# ####################################################
print('---------- RESULTS FOR TABLE -----------')
print('Collapse Scale Factor [a_col]={0}'.format(sn(a_col, 6)))
print('Numerical Non-Linear Density Contrast at Collapse [delta_inf]={0}'.format(sn(num_nonLin_dc, 6)))
print('  EdS -> delta_ini={0}, delta_crit={1}'.format(sn(delta_ini_eds, 6),
                                                      sn(delta_c_EdS, 6)))
print(' LCDM -> delta_ini={0}, delta_crit={1}'.format(sn(delta_initials_lcdm[0], 6),
                                                      sn(delta_c_LCDM, 6)))
print('LsCDM -> delta_ini={0}, delta_crit={1}'.format(sn(delta_initials_lscdm[0], 6),
                                                      sn(delta_c_LsCDM, 6)))

print('---------- RESULTS FOR NUMERICAL ANALYSIS -----------')
print('  EdS -> delta_ini={0}, delta_prime_ini={1}, delta_crit={2}'.format(delta_ini_eds,
                                                                           delta_prime_ini_eds,
                                                                           delta_c_EdS))
print(' LCDM -> delta_ini={0}, delta_prime_ini={1}, delta_crit={2}'.format(delta_initials_lcdm[0],
                                                                           delta_initials_lcdm[1],
                                                                           delta_c_LCDM))
print('LsCDM -> delta_ini={0}, delta_prime_ini={1}, delta_crit={2}'.format(delta_initials_lscdm[0],
                                                                           delta_initials_lscdm[1],
                                                                           delta_c_LsCDM))
