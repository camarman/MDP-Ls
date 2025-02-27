import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from models.lcdm import *
from models.lscdm import *
from src.calculate_growthFactor import *


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                          # initial scale factor

Om0_lcdm = Om0_finder_LCDM()          # LCDM matter density parameter

z_dag = 1.7                           # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)               # LsCDM transition scale factor
Om0_lscdm = Om0_finder_LsCDM(z_dag)   # LsCDM matter density parameter

# ----------------------------------------
# Define collapse parameters for different scenarios
collapse_params = {
    0.125: {'delta_ini_LCDM': 0.013501972203792675,
            'delta_ini_LsCDM': 0.013479340064923446},
    0.25: {'delta_ini_LCDM': 0.006786041333596292,
           'delta_ini_LsCDM': 0.006696092407629586},
    0.5: {'delta_ini_LCDM': 0.003523332881324887,
           'delta_ini_LsCDM': 0.003382970968120964},
    1.0: {'delta_ini_LCDM': 0.0021259839475166182,
          'delta_ini_LsCDM': 0.002084479946166516}
}

# Select the desired a_col
a_col = 1.0
params = collapse_params[a_col]
delta_ini_LCDM = params['delta_ini_LCDM']
delta_ini_LsCDM = params['delta_ini_LsCDM']

delta_prime_ini_LCDM = delta_ini_LCDM/a_ini
delta_prime_ini_LsCDM = delta_ini_LsCDM/a_ini

# ----------------------------------------
a_vals = np.linspace(a_ini, a_col, 1000, endpoint=True)

delta_lcdm = np.array([growthFactor_LCDM(a, a_ini, a_col, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM,
                                         delta_prime_ini=delta_prime_ini_LCDM,
                                         norm_method='unnorm') for a in a_vals])

delta_lscdm = np.array([growthFactor_LsCDM(a, a_ini, a_col, a_dag, Om0_lscdm,
                                           delta_ini=delta_ini_LsCDM,
                                           delta_prime_ini=delta_prime_ini_LsCDM,
                                           norm_method='unnorm') for a in a_vals])

delta_prime_lcdm = np.gradient(delta_lcdm, a_vals)
delta_prime_lscdm = np.gradient(delta_lscdm, a_vals)
delta_prime_ratio = delta_prime_lscdm / delta_prime_lcdm


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

ax0.plot(a_vals, delta_prime_ratio, color='#000000',
         lw=6.0, label=r"$\delta'_{\Lambda_{\rm s}} / \delta'_{\Lambda}$")
ax0.axhline(y=1, color='#117733', ls='-', lw=5.0, alpha=0.8)
ax0.axvline(x=a_col, color='#FF5733', ls=':', lw=5.0, label=rf'$a_{{\rm col}}={a_col}$')
ax0.axvline(x=a_dag, color='#BBBBBB', ls='-', lw=5.0, alpha=0.8, label=r'$a_{\dagger}$')

ax0.set_xlim(1e-2, 1.004)
ax0.set_ylim(0.80, 1.20)

ax0.set_xlabel('$a$')
ax0.set_ylabel(r"$\delta'_{\Lambda_{\rm s}} / \delta'_{\Lambda}$")

ax0.set_xscale('log')

ax0.set_xticks([1e-2, 1e-1, 1])
ax0.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20])
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

plt.tight_layout()
ax0.legend()
plt.savefig(rf'log\figure_4R_a{str(a_col).split('.')[1]}.pdf', format='pdf', dpi=2400)
plt.show()
