import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np


######################################################
# -------------------- PARAMETERS --------------------
######################################################
z_dag = 1.7               # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)   # LsCDM transition scale factor

# ----------------------------------------
# importing data
a_col = np.genfromtxt(r'log\lmdp_EdS_data.txt', skip_header=1, usemask=True, usecols=1)
delta_c_eds = np.genfromtxt(r'log\lmdp_EdS_data.txt', skip_header=1, usemask=True, usecols=4)
delta_c_lcdm = np.genfromtxt(r'log\lmdp_LCDM_data.txt', skip_header=1, usemask=True, usecols=4)
delta_c_lscdm = np.genfromtxt(r'log\lmdp_LsCDM_data.txt', skip_header=1, usemask=True, usecols=4)


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

ax0.plot(a_col, delta_c_eds, color='#DDAA33', ls='-.', lw=6.0, label=r'$\delta_{\rm c, EdS}$')
ax0.plot(a_col, delta_c_lcdm, color='#BB5566', ls='--', lw=6.0, label=r'$\delta_{{\rm c}, \Lambda}$')
ax0.plot(a_col, delta_c_lscdm, color='#004488', ls='-', lw=6.0, label=r'$\delta_{{\rm c}, \Lambda_{\rm s}}$')
ax0.axvline(x=a_dag, color='#BBBBBB', ls='-', lw=5.0, alpha=0.8, label=r'$a_{\dagger}$')

ax0.set_xlim(1e-2, 1)
ax0.set_ylim(1.676, 1.688)

ax0.set_xlabel(r'$a_{\rm col}$')
ax0.set_ylabel(r'$\delta_{\rm c}$')

ax0.set_xscale('log')

ax0.set_xticks([1e-2, 1e-1, 1])
ax0.set_yticks(np.array([1.676, 1.678, 1.68, 1.682, 1.684, 1.686, 1.688]))
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

plt.tight_layout()
ax0.legend(loc='lower left')
plt.savefig(r'log\figure_5.pdf', format='pdf', dpi=2400)
plt.show()
