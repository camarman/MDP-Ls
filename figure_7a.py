import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from models.lcdm import Om0_finder_LCDM
from models.lscdm import Om0_finder_LsCDM
from src.tools import diracDeltaApprox


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                          # initial scale factor

Om0_lcdm = Om0_finder_LCDM()          # LCDM matter density parameter

z_dag = 1.7                           # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)               # LsCDM transition scale factor
Om0_lscdm = Om0_finder_LsCDM(z_dag)   # LsCDM matter density parameter

# ----------------------------------------
def friction_LCDM(a, Om0):
    E = np.sqrt(Om0 * a ** (-3) + (1 - Om0))
    E_prime = (-3 * Om0 * a ** (-4)) / (2 * E)
    return 3 / a + E_prime / E


def friction_LsCDM(a, Om0):
    dirac_delta = diracDeltaApprox(a - a_dag)
    E = np.sqrt(Om0 * a ** (-3) + (1 - Om0) * np.sign(a - a_dag))
    E_prime = (-3 * Om0 * a ** (-4) + 2 * (1 - Om0) * dirac_delta) / (2 * E)
    return 3 / a + E_prime / E
# ----------------------------------------

a_vals = np.linspace(a_ini, 1, 50000, endpoint=True)

friction_lcdm_vals = friction_LCDM(a_vals, Om0_lcdm)
friction_lscdm_vals = friction_LsCDM(a_vals, Om0_lscdm)
percentage_F = 100 * ((friction_lscdm_vals / friction_lcdm_vals) - 1)


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

ax0.plot(a_vals, percentage_F, color='#000000', ls='-', lw=6.0, label=r'$\Delta H_f[\%]$')
ax0.axhline(y=0, color='#117733', ls='-', lw=5.0, alpha=0.8)
ax0.axvline(x=a_dag, color='#BBBBBB', ls='-', lw=5.0, alpha=0.8, label=r'$a_{\dagger}$')

ax0.set_xlim(1e-2, 1)
ax0.set_ylim(-50, +50)

ax0.set_xlabel(r'$a$')
ax0.set_ylabel(r'$\Delta H_f[\%]$')

ax0.set_xscale('log')

ax0.set_xticks([1e-2, 1e-1, 1])
ax0.set_yticks(np.arange(-50, 60, 10))
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

ax0.legend()
plt.tight_layout()
plt.savefig(r'log\figure_7a.pdf', format='pdf', dpi=2400)
plt.show()
