import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from models.lcdm import Om0_finder_LCDM
from models.lscdm import Om0_finder_LsCDM
from src.calculate_growthFactor import *
from scipy.integrate import quad


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3   # initial scale factor

a_col_1 = 1.0
a_col_2 = 0.5
a_col_3 = 0.25
a_col_4 = 0.125

Om0_lcdm = Om0_finder_LCDM()


# Initial density contrasts for LCDM
delta_ini_LCDM_1 = 0.0021259839475166182 # for a_col_1
delta_ini_LCDM_2 = 0.003523332881324887  # for a_col_2
delta_ini_LCDM_3 = 0.006786041333596292  # for a_col_3
delta_ini_LCDM_4 = 0.013501972203792675  # for a_col_4

# Initial rate of evolution for LCDM
delta_prime_ini_LCDM_1 = delta_ini_LCDM_1 / a_ini
delta_prime_ini_LCDM_2 = delta_ini_LCDM_2 / a_ini
delta_prime_ini_LCDM_3 = delta_ini_LCDM_3 / a_ini
delta_prime_ini_LCDM_4 = delta_ini_LCDM_4 / a_ini

a_vals_1 = np.linspace(a_ini, a_col_1, 3000, endpoint=True)
a_vals_2 = np.linspace(a_ini, a_col_2, 3000, endpoint=True)
a_vals_3 = np.linspace(a_ini, a_col_3, 3000, endpoint=True)
a_vals_4 = np.linspace(a_ini, a_col_4, 3000, endpoint=True)


# =============================================================
delta_lcdm_1 = np.array([growthFactor_LCDM(a, a_ini, a_col_1, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_1,
                                         delta_prime_ini=delta_prime_ini_LCDM_1,
                                         norm_method='unnorm') for a in a_vals_1])

delta_prime_lcdm_1 = np.gradient(delta_lcdm_1, a_vals_1)
f_lcdm_1 = (a_vals_1/delta_lcdm_1) * delta_prime_lcdm_1

# =============================================================

delta_lcdm_2 = np.array([growthFactor_LCDM(a, a_ini, a_col_2, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_2,
                                         delta_prime_ini=delta_prime_ini_LCDM_2,
                                         norm_method='unnorm') for a in a_vals_2])

delta_prime_lcdm_2 = np.gradient(delta_lcdm_2, a_vals_2)
f_lcdm_2 = (a_vals_2/delta_lcdm_2) * delta_prime_lcdm_2

# =============================================================

delta_lcdm_3 = np.array([growthFactor_LCDM(a, a_ini, a_col_3, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_3,
                                         delta_prime_ini=delta_prime_ini_LCDM_3,
                                         norm_method='unnorm') for a in a_vals_3])

delta_prime_lcdm_3 = np.gradient(delta_lcdm_3, a_vals_3)
f_lcdm_3 = (a_vals_3/delta_lcdm_3) * delta_prime_lcdm_3

# =============================================================

delta_lcdm_4 = np.array([growthFactor_LCDM(a, a_ini, a_col_4, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_4,
                                         delta_prime_ini=delta_prime_ini_LCDM_4,
                                         norm_method='unnorm') for a in a_vals_4])

delta_prime_lcdm_4 = np.gradient(delta_lcdm_4, a_vals_4)
f_lcdm_4 = (a_vals_4/delta_lcdm_4) * delta_prime_lcdm_4


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


ax0.plot(a_vals_1, f_lcdm_1, color="#004488", ls=':', lw=6.0,
         label=r'$\delta_{\rm ini}=2.12598\times10^{-3},\, a_{\rm col} = 1.0$')

ax0.plot(a_vals_2, f_lcdm_2, color="#BB5566", ls='-.', lw=6.0, alpha=0.9,
         label=r'$\delta_{\rm ini}=3.52333\times10^{-3},\, a_{\rm col} = 0.5$')

ax0.plot(a_vals_3, f_lcdm_3, color="#DDAA33", ls='--', lw=6.0, alpha=0.8,
         label=r'$\delta_{\rm ini}=6.78604\times10^{-3},\, a_{\rm col} = 0.25$')

ax0.plot(a_vals_4, f_lcdm_4, color="#BBBBBB", ls='-', lw=6.0, alpha=0.7,
         label=r'$\delta_{\rm ini}=1.35020\times10^{-2},\, a_{\rm col} = 0.125$')

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
plt.savefig(r'log\figure_13a.pdf', format='pdf', dpi=2400)
plt.show()

