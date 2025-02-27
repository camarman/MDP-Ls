import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from models.lcdm import *
from models.lscdm import *
from src.tools import *


######################################################
# -------------------- PARAMETERS --------------------
######################################################
Om0_lcdm = Om0_finder_LCDM()            # LCDM matter density parameter
R_lcdm = (1 - Om0_lcdm) / Om0_lcdm      # R_LCDM parameter

z_dag = 1.7                             # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)                 # LsCDM transition scale factor
Om0_lscdm = Om0_finder_LsCDM(z_dag)     # LsCDM matter density parameter
R_lscdm = (1 - Om0_lscdm) / Om0_lscdm   # R_LsCDM parameter

a_vals = np.linspace(0.1, 1, 1000000, endpoint=True)

# ----------------------------------------
def Oma_lcdm(a):
    return 1 / (1 + a**3 * R_lcdm)


def Oma_lscdm(a):
    if a < a_dag:
        return 1/(1 - a**3 * R_lscdm)
    else:
        return 1/(1 + a**3 * R_lscdm)


def f_lcdm(a):
    Oma_lcdm = 1/(1 + a**3*R_lcdm)
    return -1.5 * Oma_lcdm + 2.5 * Oma_lcdm**(3/2) * (1/sp.special.hyp2f1(5/6, 3/2, 11/6, -a**3*R_lcdm))


def f_lscdm(a):
    if a < a_dag:
        Oma_lscdm = 1 / (1 - a**3*R_lscdm)
        return -1.5 * Oma_lscdm + 2.5 * Oma_lscdm**(3/2) * (1/sp.special.hyp2f1(5/6, 3/2, 11/6, a**3*R_lscdm))
    else:
        Oma_lscdm = 1 / (1 + a**3*R_lscdm)
        return -1.5 * Oma_lscdm + 2.5 * Oma_lscdm**(3/2) * (1/sp.special.hyp2f1(5/6, 3/2, 11/6, -a**3*R_lscdm))

# -------------------- LCDM --------------------
Oma_lcdm_vals = np.array([Oma_lcdm(ai) for ai in a_vals])

# Analytical solution of the gamma parameter
f_lcdm_vals = np.array([f_lcdm(ai) for ai in a_vals])
gamma_lcdm = np.log(f_lcdm_vals) / np.log(Oma_lcdm_vals)

# Approximate solution of the gamma parameter
gamma_lcdm_approx = (6/11) + (1 - Oma_lcdm_vals) * \
    (3/125) * (2*(1+1.5) / (1+6/5)**3)

# gamma values at today
gamma_lcdm_today = gamma_lcdm[-1]
gamma_lcdm_approx_today = gamma_lcdm_approx[-1]

# -------------------- LsCDM --------------------
Oma_lscdm_vals = np.array([Oma_lscdm(ai) for ai in a_vals])

# Analytical solution of the gamma parameter
f_lscdm_vals = np.array([f_lscdm(ai) for ai in a_vals])
gamma_lscdm = np.log(f_lscdm_vals) / np.log(Oma_lscdm_vals)

# Approximate solution of the gamma parameter
gamma_lscdm_approx = (6/11) + (1 - Oma_lscdm_vals) * \
    (3/125) * (2*(1+1.5) / (1+6/5)**3)

# gamma values at today
gamma_lscdm_today = gamma_lscdm[-1]
gamma_lscdm_approx_today = gamma_lscdm_approx[-1]


####################################################
# -------------------- PLOTTING --------------------
####################################################
# https://github.com/garrettj403/SciencePlots
import scienceplots
plt.style.use(['science', 'high-vis'])

params = {
    'legend.fontsize': '45',
    'axes.labelsize':  '50',
    'axes.titlesize':  '50',
    'figure.figsize':  (18, 12),
    'xtick.labelsize': '50',
    'ytick.labelsize': '50',
    'font.family': 'serif',
    'axes.linewidth':  '3',
}
pylab.rcParams.update(params)
fig, ax0 = plt.subplots()

ax0.plot(a_vals, gamma_lcdm_approx, color='#BB5566', ls='--',
         lw=5.5, alpha=0.8, label=r'$\gamma_{\Lambda}^{(\rm approx)}$')
ax0.plot(a_vals, gamma_lscdm_approx, color='#004488', ls='--',
         lw=5.5, alpha=0.8, label=r'$\gamma_{\Lambda_{\rm s}}^{(\rm approx)}$')
ax0.plot(a_vals, gamma_lcdm, color='#BB5566', ls='-',
         lw=6.0, label=r'$\gamma_{\Lambda}$')
ax0.plot(a_vals, gamma_lscdm, color='#004488', ls='-',
         lw=6.0, label=r'$\gamma_{\Lambda_{\rm s}}$')
ax0.axhline(y=6/11, color='#DDAA33', ls='-',
            lw=5.0, label=r'$\gamma_{\rm EdS}=6/11$')

print('Gamma Values Today:')
print('Theoretical:')
print('gamma LCDM:{0}\ngamma LsCDM :{1}'.format(round(gamma_lcdm_today, 3),
                                                round(gamma_lscdm_today, 3)))
print('Approximation:')
print('gamma LCDM:{0}\ngamma LsCDM:{1}'.format(round(gamma_lcdm_approx_today, 3),
                                               round(gamma_lscdm_approx_today, 3)))

ax0.set_xlim(0.1, 1)
ax0.set_ylim(0.543, 0.557)

ax0.set_xlabel('$a$')
ax0.set_ylabel(r'$\gamma$')

ax0.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax0.set_yticks([0.543, 0.545, 0.547, 0.549, 0.551, 0.553, 0.555, 0.557])
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

plt.tight_layout()
ax0.legend(loc='upper left')
plt.savefig(r'log/figure_11.pdf', format='pdf', dpi=2400)
plt.show()
