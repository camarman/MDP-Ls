import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from models.lcdm import Om0_finder_LCDM
from models.lscdm import Om0_finder_LsCDM
from src.calculate_growthFactor import *
from src.calculate_growthRate import *
from src.tools import sn


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                          # initial scale factor
a_col = 1                             # collapse scale factor

Om0_lcdm = Om0_finder_LCDM()            # LCDM matter density parameter
R_lcdm = (1 - Om0_lcdm) / Om0_lcdm      # R_LCDM parameter

z_dag = 1.7                             # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)                 # LsCDM transition scale factor
Om0_lscdm = Om0_finder_LsCDM(z_dag)     # LsCDM matter density parameter
R_lscdm = (1 - Om0_lscdm) / Om0_lscdm   # R_LsCDM parameter


# Initial density contrasts for EdS, LCDM, and LsCDM Models
delta_ini_EdS = 0.0016864701998411454
delta_ini_LCDM = 0.0021259839475166182
delta_ini_LsCDM = 0.002084479946166516

# Initial rate of evolution for EdS, LCDM, and LsCDM Models
delta_prime_ini_EdS = delta_ini_EdS / a_ini
delta_prime_ini_LCDM = delta_ini_LCDM / a_ini
delta_prime_ini_LsCDM = delta_ini_LsCDM / a_ini

# ----------------------------------------
a_vals = np.linspace(a_ini, a_col, 1000, endpoint=True)

# ---------------------- DIRECT METHOD ----------------------
f_eds_numerical_direct = np.array([growthRate_EdS(a, a_ini, a_col, f_ini=1) for a in a_vals])
f_lcdm_numerical_direct  = np.array([growthRate_LCDM(a, a_ini, a_col, Om0_lcdm, f_ini=1) for a in a_vals])
f_lscdm_numerical_direct  = np.array([growthRate_LsCDM(a, a_ini, a_col, a_dag, Om0_lscdm, f_ini=1) for a in a_vals])

# ---------------------- IN-DIRECT METHOD ----------------------
delta_eds = np.array([growthFactor_EdS(a, a_ini, a_col,
                                       delta_ini=delta_ini_EdS,
                                       delta_prime_ini=delta_prime_ini_EdS,
                                       norm_method='unnorm') for a in a_vals])

delta_lcdm = np.array([growthFactor_LCDM(a, a_ini, a_col, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM,
                                         delta_prime_ini=delta_prime_ini_LCDM,
                                         norm_method='unnorm') for a in a_vals])

delta_lscdm = np.array([growthFactor_LsCDM(a, a_ini, a_col, a_dag, Om0_lscdm,
                                           delta_ini=delta_ini_LsCDM,
                                           delta_prime_ini=delta_prime_ini_LsCDM,
                                           norm_method='unnorm') for a in a_vals])

delta_prime_eds = np.gradient(delta_eds, a_vals)
delta_prime_lcdm = np.gradient(delta_lcdm, a_vals)
delta_prime_lscdm = np.gradient(delta_lscdm, a_vals)

f_eds_numerical_indirect = (a_vals / delta_eds) * delta_prime_eds
f_lcdm_numerical_indirect = (a_vals / delta_lcdm) * delta_prime_lcdm
f_lscdm_numerical_indirect = (a_vals / delta_lscdm) * delta_prime_lscdm


####################################################
# -------------------- PLOTTING --------------------
####################################################
# https://github.com/garrettj403/SciencePlots
import scienceplots
plt.style.use(['science', 'high-vis'])

params = {
    'legend.fontsize': '44',
    'axes.labelsize':  '50',
    'figure.figsize':  (18, 12),
    'xtick.labelsize': '50',
    'ytick.labelsize': '50',
    'font.family': 'serif',
    'axes.linewidth':  '3',
}
pylab.rcParams.update(params)
fig, ax0 = plt.subplots()

ax0.plot(a_vals, f_eds_numerical_direct, color='#DDAA33',
         ls='-', lw=6.0, alpha=0.4, label=r'$f_{\rm EdS}$ (with $f_{\rm ini}$)')
ax0.plot(a_vals, f_eds_numerical_indirect, color='#DDAA33',
         ls='--', lw=6.0, label=r'$f_{\rm EdS}$ (with $\delta_{\rm ini}$)')
ax0.plot(a_vals, f_lcdm_numerical_direct, color='#BB5566',
         ls='-', lw=6.0, alpha=0.4, label=r'$f_{\Lambda}$ (with $f_{\rm ini}$)')
ax0.plot(a_vals, f_lcdm_numerical_indirect, color='#BB5566',
         ls='--', lw=6.0, label=r'$f_{\Lambda}$ (with $\delta_{\rm ini}$)')
ax0.plot(a_vals, f_lscdm_numerical_direct, color='#004488',
         ls='-', lw=6.0, alpha=0.4, label=r'$f_{\Lambda_{\rm s}}$ (with $f_{\rm ini}$)')
ax0.plot(a_vals, f_lscdm_numerical_indirect, color='#004488',
         ls='--', lw=6.0, label=r'$f_{\Lambda_{\rm s}}$ (with $\delta_{\rm ini}$)')


ax0.set_xlim(1e-2, 1)
ax0.set_ylim(0.4, 1.2)

ax0.set_xlabel('$a$')
ax0.set_ylabel(r'$f$')

ax0.set_xscale('log')

ax0.set_xticks([1e-2, 1e-1, 1])
ax0.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

plt.tight_layout()
plt.legend(loc='lower left')
plt.savefig(r'log\figure_8.pdf', format='pdf', dpi=2400)
plt.show()
