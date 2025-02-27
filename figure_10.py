import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from models.lcdm import *
from models.lscdm import *
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

# ----------------------------------------
def f_EdS(a):
    return 1


def f_lcdm(a):
    Oma_lcdm = 1/(1 + a**3*R_lcdm)
    return -1.5 * Oma_lcdm + 2.5 * Oma_lcdm**(3/2) * (1/sp.special.hyp2f1(5/6, 3/2, 11/6, -a**3*R_lcdm))


def f_lscdm_AdS(a):
    Oma_lscdm = 1 / (1 - a**3*R_lscdm)
    return -1.5 * Oma_lscdm + 2.5 * Oma_lscdm**(3/2) * (1/sp.special.hyp2f1(5/6, 3/2, 11/6, a**3*R_lscdm))


def f_lscdm_dS(a):
    Oma_lscdm = 1 / (1 + a**3*R_lscdm)
    return -1.5 * Oma_lscdm + 2.5 * Oma_lscdm**(3/2) * (1/sp.special.hyp2f1(5/6, 3/2, 11/6, -a**3*R_lscdm))
# ------------------------------------------------------------
a_vals = np.logspace(np.log10(a_ini), np.log10(a_col), num=1000, endpoint=True)

a_vals_AdS = np.array([a_i for a_i in a_vals if a_i<=a_dag])
a_vals_dS = np.array([a_i for a_i in a_vals if a_i>=a_dag])

f_eds_numerical = np.array([growthRate_EdS(a, a_ini, a_col, f_ini=1) for a in a_vals])
f_lcdm_numerical = np.array([growthRate_LCDM(a, a_ini, a_col, Om0_lcdm, f_ini=1) for a in a_vals])
f_lscdm_numerical = np.array([growthRate_LsCDM(a, a_ini, a_col, a_dag, Om0_lscdm, f_ini=1) for a in a_vals])

f_eds_analytical = np.array([f_EdS(ai) for ai in a_vals])
f_lcdm_analytical = np.array([f_lcdm(ai) for ai in a_vals])
f_lscdm_analytical_AdS = np.array([f_lscdm_AdS(ai) for ai in a_vals_AdS])
f_lscdm_analytical_dS = np.array([f_lscdm_dS(ai) for ai in a_vals_dS])
# ------------------------------------------------------------

term_c1 = 5 / (np.sqrt(1+a_ini**3*R_lcdm) * (5 + 2*a_ini**3*R_lcdm))
c1 = (2/5) * a_ini**(5/2) * (term_c1 - sp.special.hyp2f1(5/6, 3/2, 11/6, -a_ini**3*R_lcdm))

term_c2 = 5 / (np.sqrt(1-a_ini**3*R_lscdm) * (5 - 2*a_ini**3*R_lscdm))
c2 = (2/5) * a_ini**(5/2) * (term_c2 - sp.special.hyp2f1(5/6, 3/2, 11/6, a_ini**3*R_lscdm))

term_c3 = 5 / (np.sqrt(1+a_ini**3*R_lscdm) * (5 + 2*a_ini**3*R_lscdm))
c3 = (2/5) * a_ini**(5/2) * (term_c3 - sp.special.hyp2f1(5/6, 3/2, 11/6, -a_ini**3*R_lscdm))

print('C1:{}'.format(sn(c1, 1)))
print('C2:{}'.format(sn(c2, 1)))
print('C3:{}'.format(sn(c3, 1)))


####################################################
# -------------------- PLOTTING --------------------
####################################################
# https://github.com/garrettj403/SciencePlots
import scienceplots
plt.style.use(['science', 'high-vis'])

params = {
    'legend.fontsize': '40',
    'axes.labelsize':  '50',
    'figure.figsize':  (18, 12),
    'xtick.labelsize': '50',
    'ytick.labelsize': '50',
    'font.family': 'serif',
    'axes.linewidth':  '3',
}
pylab.rcParams.update(params)
fig, ax0 = plt.subplots()

ax0.plot(a_vals, f_eds_numerical, color='#DDAA33',
         ls='-', lw=6.0, alpha=0.60, label=r'$f_{\rm EdS}$ (Numerical)')
ax0.plot(a_vals, f_eds_analytical, color='#DDAA33',
         ls=':', lw=6.0, label=r'$f_{\rm EdS}$ (Analytical)')
ax0.plot(a_vals, f_lcdm_numerical, color='#BB5566',
         ls='-', lw=6.0, alpha=0.60, label=r'$f_{\Lambda}$ (Numerical)')
ax0.plot(a_vals, f_lcdm_analytical, color='#BB5566',
         ls='-.', lw=6.0, label=r'$f_{\Lambda}$ (Analytical)')

ax0.plot(a_vals, f_lscdm_numerical, color='#004488',
         ls='-', lw=6.0, alpha=0.60, label=r'$f_{\Lambda_{\rm s}}$ (Numerical)')

ax0.plot(a_vals_AdS, f_lscdm_analytical_AdS, color='#004488',
         ls='--', lw=6.0, label=r'$f_{\Lambda_{\rm s}}$ (Analytical)')
ax0.plot(a_vals_dS, f_lscdm_analytical_dS, color='#004488',
         ls='--', lw=6.0, label=r'$f_{\Lambda_{\rm s}}$ (Analytical)')

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

box = ax0.get_position()
plt.legend(loc='lower left')
plt.savefig(r'log\figure_10.pdf', format='pdf', dpi=2400)
plt.show()
