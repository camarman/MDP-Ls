import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from src.calculate_delta_inf import delta_inf


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3   # initial scale factor
a_col = 1      # collapse scale factor

# ----------------------------------------
a_col_vals = np.logspace(np.log10(1e-2), np.log10(a_col), num=1000, endpoint=True)
delta_inf_vals = np.array([delta_inf(a_ini, a_col_i) for a_col_i in a_col_vals])


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

ax0.plot(a_col_vals, delta_inf_vals, color='#000000', ls='-', lw=6.0)

ax0.set_xlim(1e-2, 1)

ax0.set_xlabel(r'$a_{\rm col}$')
ax0.set_ylabel(r'$\delta_{\infty} \equiv \delta_{\infty, {\rm EdS}}$')

ax0.set_yscale('log')
ax0.set_xscale('log')

ax0.set_xticks([1e-2, 1e-1, 1])
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

plt.tight_layout()
plt.savefig(r'log\figure_2.pdf', format='pdf', dpi=2400)
plt.show()
