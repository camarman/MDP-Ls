import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from models.lcdm import Om0_finder_LCDM
from models.lscdm import Om0_finder_LsCDM
from src.calculate_col_params_lscdm import *
from src.calculate_critical_densityContrast import crit_densityContrast_LsCDM
from src.calculate_growthFactor import *
from src.tools import sn


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                           # initial scale factor

Om0_lcdm = Om0_finder_LCDM()           # LCDM matter density parameter

z_dag = 1.7                            # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)                # LsCDM transition scale factor
Om0_lscdm = Om0_finder_LsCDM(z_dag)    # LsCDM matter density parameter
R_lscdm = (1 - Om0_lscdm) / Om0_lscdm  # R_LsCDM parameter

# ----------------------------------------
collapse_params = {
    0.125: {'delta_ini': 0.013501972203792675},
    0.25: {'delta_ini': 0.006786041333596292},
    0.5: {'delta_ini': 0.003523332881324887},
    1.0: {'delta_ini': 0.0021259839475166182}
}

# Select the desired a_col
a_col_LCDM = 1.0
params = collapse_params[a_col_LCDM]
delta_ini = params['delta_ini']

delta_prime_ini = delta_ini / a_ini
a_end = a_col_LCDM
# ----------------------------------------
a_col_LsCDM, num_nonLin_dc_LsCDM = col_params_LsCDM(a_ini, a_dag, a_end,
                                                    R_lscdm, delta_ini, delta_prime_ini)

delta_c_LsCDM = crit_densityContrast_LsCDM(a_ini, a_col_LsCDM, a_dag, Om0_lscdm, delta_ini, delta_prime_ini)

a_vals_lscdm = np.linspace(a_ini, a_col_LsCDM, 1000, endpoint=True)

delta_lcdm_vals = np.array([growthFactor_LCDM(a_i, a_ini, a_col_LCDM, Om0_lcdm,
                                              delta_ini, delta_prime_ini, norm_method='unnorm') for a_i in a_vals_lscdm])

delta_lscdm_vals = np.array([growthFactor_LsCDM(a_i, a_ini, a_col_LsCDM, a_dag, Om0_lscdm,
                                           delta_ini, delta_prime_ini, norm_method='unnorm') for a_i in a_vals_lscdm])


delta_prime_lcdm = np.gradient(delta_lcdm_vals, a_vals_lscdm)
delta_prime_lscdm = np.gradient(delta_lscdm_vals, a_vals_lscdm)
delta_prime_ratio = delta_prime_lscdm / delta_prime_lcdm

print('a_col_LsCDM = {}'.format(sn(a_col_LsCDM, 4)))
print('num_nonLin_dc_LsCDM = {}'.format(sn(num_nonLin_dc_LsCDM, 6)))
print('delta_c_LsCDM({0}) = {1}'.format(sn(a_col_LsCDM, 4), sn(delta_c_LsCDM, 6)))

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

ax0.plot(a_vals_lscdm, delta_prime_ratio, color='#000000', ls='-',
        lw=6.0, label=r"$\delta'_{\Lambda_{\rm s}}/ \delta'_{\Lambda}$")

ax0.axhline(y=1, color='#117733', ls='-', lw=5.0, alpha=0.8)
ax0.axvline(x=a_col_LsCDM, color='#FF5733', ls=':', lw=5.0, label=rf'$a_{{\rm col, \Lambda_{{\rm s}}}}={round(a_col_LsCDM, 4)}$')
ax0.axvline(x=a_dag, color='#BBBBBB', ls='-', lw=5.0, alpha=0.8, label=r'$a_{\dagger}$')

ax0.set_xlim(0.01, 1)
ax0.set_ylim(0.8, 1.2)

ax0.set_xlabel('$a$')
ax0.set_ylabel(r"$\delta'_{\Lambda_{\rm s}}/ \delta'_{\Lambda}~[a \leq a_{\rm col,\Lambda_{\rm s}}]$")

ax0.set_xscale('log')

ax0.set_xticks([1e-2, 1e-1, 1])
ax0.set_yticks([0.80, 0.90, 1.0, 1.1, 1.2])
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

ax0.legend(loc='upper left')
plt.tight_layout()
plt.savefig(rf'log\figure_6R_a{str(a_col_LCDM).split('.')[1]}.pdf', format='pdf', dpi=2400)
plt.show()
