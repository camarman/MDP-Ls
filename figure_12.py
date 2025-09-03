import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from models.lcdm import Om0_finder_LCDM
from models.lscdm import Om0_finder_LsCDM
from src.calculate_growthFactor import *


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                          # initial scale factor
a_col = 1                             # collapse scale factor

Om0_lcdm = Om0_finder_LCDM()          # LCDM matter density parameter

z_dag = 1.7                           # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)               # LsCDM transition scale factor
Om0_lscdm = Om0_finder_LsCDM(z_dag)   # LsCDM matter density parameter

# Initial density contrasts for LCDM, and LsCDM Models
delta_ini_LCDM = 0.0021259839475166182
delta_ini_LsCDM = 0.002084479946166516

# Initial rate of evolution for LCDM, and LsCDM Models
delta_prime_ini_LCDM = delta_ini_LCDM / a_ini
delta_prime_ini_LsCDM = delta_ini_LsCDM / a_ini

sigma8_lcdm = 0.816
sigma8_lscdm = 0.809

# =============================================================
a_vals = np.linspace(a_ini, a_col, 5000, endpoint=True)
z_vals = 1 / a_vals - 1

delta_lcdm = np.array([growthFactor_LCDM(a, a_ini, a_col, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM,
                                         delta_prime_ini=delta_prime_ini_LCDM,
                                         norm_method='unnorm') for a in a_vals])

delta_lscdm = np.array([growthFactor_LsCDM(a, a_ini, a_col, a_dag, Om0_lscdm,
                                           delta_ini=delta_ini_LsCDM,
                                           delta_prime_ini=delta_prime_ini_LsCDM,
                                           norm_method='unnorm') for a in a_vals])

# =============================================================
growth_factor_LCDM = np.array([growthFactor_LCDM(a, a_ini, a_col, Om0_lcdm,
                                                 delta_ini=delta_ini_LCDM,
                                                 delta_prime_ini=delta_prime_ini_LCDM,
                                                 norm_method='norm') for a in a_vals])


growth_factor_LsCDM = np.array([growthFactor_LsCDM(a, a_ini, a_col, a_dag, Om0_lscdm,
                                                   delta_ini=delta_ini_LsCDM,
                                                   delta_prime_ini=delta_prime_ini_LsCDM,
                                                   norm_method='norm') for a in a_vals])


delta_prime_lcdm = np.gradient(delta_lcdm, a_vals)
delta_prime_lscdm = np.gradient(delta_lscdm, a_vals)

fs8_lcdm = (a_vals/delta_lcdm) * delta_prime_lcdm * sigma8_lcdm * growth_factor_LCDM
fs8_lscdm = (a_vals/delta_lscdm) * delta_prime_lscdm * sigma8_lscdm * growth_factor_LsCDM


####################################################
# -------------------- PLOTTING --------------------
####################################################
# https://github.com/garrettj403/SciencePlots
import scienceplots
plt.style.use(['science', 'high-vis'])

params = {
    'legend.fontsize': '26',
    'axes.labelsize':  '50',
    'figure.figsize':  (18, 12),
    'xtick.labelsize': '50',
    'ytick.labelsize': '50',
    'font.family': 'serif',
    'axes.linewidth':  '3',
}
pylab.rcParams.update(params)
fig, ax0 = plt.subplots()

ax0.plot(z_vals, fs8_lcdm, color='#BB5566', ls='--', lw=6.0, label=r'$f\sigma_{8}$ ($\Lambda$CDM)')
ax0.plot(z_vals, fs8_lscdm, color='#004488', ls='-', lw=6.0, label=r'$f\sigma_{8}$ ($\Lambda_{\rm s}$CDM)')

# =========================== Data points ===========================
# Data from the table
z = np.genfromtxt(r'log\fs8_table.txt', skip_header=1, usemask=True, usecols=1)
f_sigma8 = np.genfromtxt(r'log\fs8_table.txt', skip_header=1, usemask=True, usecols=2)
errors = np.genfromtxt(r'log\fs8_table.txt', skip_header=1, usemask=True, usecols=3)

# Labels for datasets
labels = [
    'SnIa IRAS', '2MRS', '6dFGS+SnIa', 'SDSS-veloc', 'SDSS-MGS', '2dFGRS', 'GAMA',
    'SDSS-LRG-200', 'BOSS LOWZ', 'SDSS-LRG-200', 'GAMA', 'WiggleZ', 'SDSS-CMASS',
    'WiggleZ', 'Vipers PDR-2', 'WiggleZ', 'Vipers PDR-2', 'SDSS-IV eBOSS', 'SDSS-IV eBOSS',
    'FastSound', 'SDSS-IV eBOSS', 'SDSS-IV eBOSS'
]

# Mapping unique datasets to color and marker
dataset_mapping = {
    'SnIa IRAS': {'color': 'blue', 'marker': 'o'},
    '2MRS': {'color': 'green', 'marker': 's'},
    '6dFGS+SnIa': {'color': 'orange', 'marker': '^'},
    'SDSS-veloc': {'color': 'purple', 'marker': 'D'},
    'SDSS-MGS': {'color': 'brown', 'marker': 'v'},
    '2dFGRS': {'color': 'pink', 'marker': 'P'},
    'GAMA': {'color': 'gray', 'marker': 'X'},
    'SDSS-LRG-200': {'color': 'cyan', 'marker': '*'},
    'BOSS LOWZ': {'color': 'magenta', 'marker': 'p'},
    'WiggleZ': {'color': 'red', 'marker': 'H'},
    'SDSS-CMASS': {'color': 'yellow', 'marker': '<'},
    'Vipers PDR-2': {'color': 'lime', 'marker': '>'},
    'SDSS-IV eBOSS': {'color': 'teal', 'marker': '+'},
    'FastSound': {'color': 'navy', 'marker': 'h'}
}

# Plot each data point, using the same color and marker for identical labels
for i in range(len(z)):
    dataset = labels[i]
    color_marker = dataset_mapping.get(
        dataset, {'color': 'black', 'marker': 'o'})
    plt.errorbar(z[i], f_sigma8[i], yerr=errors[i], fmt=color_marker['marker'],
                 ecolor=color_marker['color'], color=color_marker['color'],
                 elinewidth=4, markersize=10, capsize=4, capthick=4, label=dataset)

ax0.axvline(x=z_dag, color='#BBBBBB', ls='-', lw=5.0, alpha=0.8, label=r'$a_{\dagger}$')
ax0.set_xlim(0, 2)
ax0.set_ylim(0.2, 0.9)

ax0.set_xlabel('$z$')
ax0.set_ylabel(r'$f\sigma_8$')

ax0.set_xticks([0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
ax0.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

handles, unique_labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(unique_labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4)
plt.tight_layout()
plt.savefig(r'log\figure_12.pdf', format='pdf', dpi=2400)
plt.show()
