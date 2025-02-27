import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from models.lcdm import *
from models.lscdm import *


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                          # initial scale factor

Om0_lcdm = Om0_finder_LCDM()          # LCDM matter density parameter

z_dag = 1.7                           # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)               # LsCDM transition scale factor
Om0_lscdm = Om0_finder_LsCDM(z_dag)   # LsCDM matter density parameter

# ----------------------------------------
def potential_lcdm(a, Om0):
    E = np.sqrt(Om0 * a ** (-3) + (1 - Om0))
    Om = (Om0 * a ** (-3)) / E ** 2
    return -1.5 * Om / a ** 2


def potential_lscdm(a, Om0):
    E = np.sqrt(Om0 * a ** (-3) + (1 - Om0) * np.sign(a - a_dag))
    Om = (Om0 * a ** (-3)) / E ** 2
    return -1.5 * Om / a ** 2
# ----------------------------------------

a_vals = np.linspace(a_ini, 1, 50000, endpoint=True)

potential_lcdm_vals = potential_lcdm(a_vals, Om0_lcdm)
potential_lscdm_vals = potential_lscdm(a_vals, Om0_lscdm)
percentage_P = 100 * ((potential_lscdm_vals / potential_lcdm_vals) - 1)


####################################################
# -------------------- PLOTTING --------------------
####################################################
# https://github.com/garrettj403/SciencePlots
import scienceplots
plt.style.use(['science', 'high-vis'])

params = {
    'legend.fontsize': '45',
    'axes.labelsize':  '50',
    'figure.figsize':  (18, 12),
    'xtick.labelsize': '50',
    'ytick.labelsize': '50',
    'font.family': 'serif',
    'axes.linewidth':  '3',
}
pylab.rcParams.update(params)
fig, ax0 = plt.subplots()

ax0.plot(a_vals, percentage_P, color='#000000', ls='-', lw=6.0, label=r'$\Delta \Phi[\%]$')
ax0.axhline(y=0, color='#117733', ls='-', lw=5.0, alpha=0.8)
ax0.axvline(x=a_dag, color='#BBBBBB', ls='-', lw=5.0, alpha=0.8, label=r'$a_{\dagger}$')

ax0.set_xlim(1e-2, 1)
ax0.set_ylim(-30, +30)

ax0.set_xlabel('$a$')
ax0.set_ylabel(r'$\Delta \Phi[\%]$')

ax0.set_xscale('log')

ax0.set_xticks([1e-2, 1e-1, 1])
ax0.set_yticks(np.arange(-30, 40, 10))
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

ax0.legend()
plt.tight_layout()
plt.savefig(r'log\figure_7b.pdf', format='pdf', dpi=2400)
plt.show()
