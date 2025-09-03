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
a_col_2 = 0.95
a_col_3 = 0.90
a_col_4 = 0.85

Om0_lcdm = Om0_finder_LCDM()
sigma8_lcdm = 0.816

# Initial density contrasts for LCDM
delta_ini_LCDM_1 = 0.0021259839475166182 # for a_col_1
delta_ini_LCDM_2 = 0.002187377134426234  # for a_col_2
delta_ini_LCDM_3 = 0.0022578657497208383  # for a_col_3
delta_ini_LCDM_4 = 0.0023392831216687616   # for a_col_4

# Initial rate of evolution for LCDM
delta_prime_ini_LCDM_1 = delta_ini_LCDM_1 / a_ini
delta_prime_ini_LCDM_2 = delta_ini_LCDM_2 / a_ini
delta_prime_ini_LCDM_3 = delta_ini_LCDM_3 / a_ini
delta_prime_ini_LCDM_4 = delta_ini_LCDM_4 / a_ini

a_vals_1 = np.linspace(a_ini, a_col_1, 500, endpoint=True)
a_vals_2 = np.linspace(a_ini, a_col_2, 500, endpoint=True)
a_vals_3 = np.linspace(a_ini, a_col_3, 500, endpoint=True)
a_vals_4 = np.linspace(a_ini, a_col_4, 500, endpoint=True)

z_vals_1 = 1 / a_vals_1 - 1
z_vals_2 = 1 / a_vals_2 - 1
z_vals_3 = 1 / a_vals_3 - 1
z_vals_4 = 1 / a_vals_4 - 1


# =============================================================
delta_lcdm_1 = np.array([growthFactor_LCDM(a, a_ini, a_col_1, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_1,
                                         delta_prime_ini=delta_prime_ini_LCDM_1,
                                         norm_method='unnorm') for a in a_vals_1])

growth_factor_LCDM_1 = np.array([growthFactor_LCDM(a, a_ini, a_col_1, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_1,
                                         delta_prime_ini=delta_prime_ini_LCDM_1,
                                         norm_method='norm') for a in a_vals_1])

delta_prime_lcdm_1 = np.gradient(delta_lcdm_1, a_vals_1)
f_lcdm_1 = (a_vals_1/delta_lcdm_1) * delta_prime_lcdm_1
sigma8_1 = sigma8_lcdm * growth_factor_LCDM_1
fs8_1 = f_lcdm_1 * sigma8_1

# =============================================================

delta_lcdm_2 = np.array([growthFactor_LCDM(a, a_ini, a_col_2, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_2,
                                         delta_prime_ini=delta_prime_ini_LCDM_2,
                                         norm_method='unnorm') for a in a_vals_2])

growth_factor_LCDM_2 = np.array([growthFactor_LCDM(a, a_ini, a_col_2, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_2,
                                         delta_prime_ini=delta_prime_ini_LCDM_2,
                                         norm_method='norm') for a in a_vals_2])

delta_prime_lcdm_2 = np.gradient(delta_lcdm_2, a_vals_2)
f_lcdm_2 = (a_vals_2/delta_lcdm_2) * delta_prime_lcdm_2
sigma8_2 = sigma8_lcdm * growth_factor_LCDM_2
fs8_2 = f_lcdm_2 * sigma8_2

# =============================================================

delta_lcdm_3 = np.array([growthFactor_LCDM(a, a_ini, a_col_3, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_3,
                                         delta_prime_ini=delta_prime_ini_LCDM_3,
                                         norm_method='unnorm') for a in a_vals_3])


growth_factor_LCDM_3 = np.array([growthFactor_LCDM(a, a_ini, a_col_3, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_3,
                                         delta_prime_ini=delta_prime_ini_LCDM_3,
                                         norm_method='norm') for a in a_vals_3])

delta_prime_lcdm_3 = np.gradient(delta_lcdm_3, a_vals_3)
f_lcdm_3 = (a_vals_3/delta_lcdm_3) * delta_prime_lcdm_3
sigma8_3 = sigma8_lcdm * growth_factor_LCDM_3
fs8_3 = f_lcdm_3 * sigma8_3


# =============================================================

delta_lcdm_4 = np.array([growthFactor_LCDM(a, a_ini, a_col_4, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_4,
                                         delta_prime_ini=delta_prime_ini_LCDM_4,
                                         norm_method='unnorm') for a in a_vals_4])

growth_factor_LCDM_4 = np.array([growthFactor_LCDM(a, a_ini, a_col_4, Om0_lcdm,
                                         delta_ini=delta_ini_LCDM_4,
                                         delta_prime_ini=delta_prime_ini_LCDM_4,
                                         norm_method='norm') for a in a_vals_4])

delta_prime_lcdm_4 = np.gradient(delta_lcdm_4, a_vals_4)
f_lcdm_4 = (a_vals_4/delta_lcdm_4) * delta_prime_lcdm_4
sigma8_4 = sigma8_lcdm * growth_factor_LCDM_4
fs8_4 = f_lcdm_4 * sigma8_4


####################################################
# -------------------- PLOTTING --------------------
####################################################
# https://github.com/garrettj403/SciencePlots
import scienceplots
plt.style.use(['science', 'high-vis'])

params = {
    'legend.fontsize': '27',
    'axes.labelsize':  '50',
    'figure.figsize':  (18, 12),
    'xtick.labelsize': '50',
    'ytick.labelsize': '50',
    'font.family': 'serif',
    'axes.linewidth':  '3',
}
pylab.rcParams.update(params)
fig, ax0 = plt.subplots()

ax0.plot(z_vals_1, fs8_1, color='#004488', ls='-', lw=6.0, label=r'$a_{\rm col} = 1.0$')
ax0.plot(z_vals_2, fs8_2, color='#BB5566', ls=':', lw=6.0, label=r'$a_{\rm col}=0.95$')
ax0.plot(z_vals_3, fs8_3, color="#DDAA33", ls='-.', lw=6.0, label=r'$a_{\rm col}=0.90$')
ax0.plot(z_vals_4, fs8_4, color="#BBBBBB", ls='--', lw=6.0, label=r'$a_{\rm col}=0.85$')

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

ax0.set_xlim(0, 2)
ax0.set_ylim(0.2, 0.9)

ax0.set_xlabel('$z$')
ax0.set_ylabel(r'$f_{\Lambda}\sigma_{8,\Lambda}$')

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
plt.savefig(r'log\figure_13b.pdf', format='pdf', dpi=2400)
plt.show()