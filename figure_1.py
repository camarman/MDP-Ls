import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import scipy as sp

from models.lcdm import Om0_finder_LCDM
from models.lscdm import Om0_finder_LsCDM
from src.tools import diracDeltaApprox


######################################################
# -------------------- PARAMETERS --------------------
######################################################
a_ini = 1e-3                          # initial scale factor
a_col = 1                             # collapse scale factor

Om0_lcdm = Om0_finder_LCDM()          # LCDM matter density parameter

z_dag = 1.7                           # LsCDM transition redshift
a_dag = 1 / (1 + z_dag)               # LsCDM transition scale factor
Om0_lscdm = Om0_finder_LsCDM(z_dag)   # LsCDM matter density parameter

# Initial density contrasts for EdS, LCDM, and LsCDM Models
delta_ini_EdS = 0.0016864701998411454
delta_ini_LCDM = 0.0021259839475166182
delta_ini_LsCDM = 0.002084479946166516

# Initial rate of evolution for EdS, LCDM, and LsCDM Models
delta_prime_ini_EdS = delta_ini_EdS / a_ini
delta_prime_ini_LCDM = delta_ini_LCDM / a_ini
delta_prime_ini_LsCDM = delta_ini_LsCDM / a_ini

##########################################################################
# ---------- NON-LINEAR AND LINEAR MATTER DENSITY PERTURBATIONS ----------
##########################################################################

# =============== EdS (Non-Linear Matter Density Perturbations)
def non_lin_density_perturbation_EdS(a_eval, delta_ini, delta_prime_ini):

    def nonLin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        E = np.sqrt(a ** (-3))
        E_prime = (-3 * a ** (-4)) / (2 * E)

        Om = (a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 4 / 3
        c3 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * (delta_prime ** 2 / (1 + delta)) + c3 * delta * (1 + delta)]

    res = sp.integrate.solve_ivp(nonLin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                method='Radau', t_eval=[a_eval], atol=1e-8, rtol=1e-8)

    if res['success']:
        deltaMatter_nonLin = res['y'][0, 0]
        return deltaMatter_nonLin

# =============== EdS (Linear Matter Density Perturbations)
def lin_density_perturbation_EdS(a_eval, delta_ini, delta_prime_ini):

    def lin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        E = np.sqrt(a ** (-3))
        E_prime = (-3 * a ** (-4)) / (2 * E)

        Om = (a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * delta]

    res = sp.integrate.solve_ivp(lin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                method='Radau', t_eval=[a_eval], atol=1e-8, rtol=1e-8)
    return res['y'][0, 0]

# =============== LCDM (Non-Linear Matter Density Perturbations)
def non_lin_density_perturbation_LCDM(a_eval, Om0, delta_ini, delta_prime_ini):

    def nonLin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0))
        E_prime = (-3 * Om0 * a ** (-4)) / (2 * E)

        Om = (Om0 * a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 4 / 3
        c3 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * (delta_prime ** 2 / (1 + delta)) + c3 * delta * (1 + delta)]

    # initial search parameter
    res = sp.integrate.solve_ivp(nonLin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                method='Radau', t_eval=[a_eval], atol=1e-8, rtol=1e-8)

    if res['success']:
        deltaMatter_nonLin = res['y'][0, 0]
        return deltaMatter_nonLin

# =============== LCDM (Linear Matter Density Perturbations)
def lin_density_perturbation_LCDM(a_eval, Om0, delta_ini, delta_prime_ini):

    def lin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0))
        E_prime = (-3 * Om0 * a ** (-4)) / (2 * E)

        Om = (Om0 * a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * delta]

    res = sp.integrate.solve_ivp(lin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                method='Radau', t_eval=[a_eval], atol=1e-8, rtol=1e-8)
    return res['y'][0, 0]

# =============== LsCDM (Non-Linear Matter Density Perturbations)
def non_lin_density_perturbation_LsCDM(a_eval, Om0, delta_ini, delta_prime_ini):

    def nonLin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        dirac_delta = diracDeltaApprox(a - a_dag)
        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0) * np.sign(a - a_dag))
        E_prime = (-3 * Om0 * a ** (-4) + (1 - Om0) * 2 * dirac_delta) / (2 * E)

        Om = (Om0 * a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 4 / 3
        c3 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * (delta_prime ** 2 / (1 + delta)) + c3 * delta * (1 + delta)]

    # initial search parameter
    res = sp.integrate.solve_ivp(nonLin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                method='Radau', t_eval=[a_eval], atol=1e-8, rtol=1e-8)

    if res['success']:
        deltaMatter_nonLin = res['y'][0, 0]
        return deltaMatter_nonLin

# =============== LsCDM (Linear Matter Density Perturbations)
def lin_density_perturbation_LsCDM(a_eval, Om0, delta_ini, delta_prime_ini):

    def lin_mdp(a, d):
        delta = d[0]
        delta_prime = d[1]

        dirac_delta = diracDeltaApprox(a - a_dag)
        E = np.sqrt(Om0 * a ** (-3) + (1 - Om0) * np.sign(a - a_dag))
        E_prime = (-3 * Om0 * a ** (-4) + (1 - Om0) * 2 * dirac_delta) / (2 * E)

        Om = (Om0 * a ** (-3)) / E ** 2
        c1 = -(3 / a + E_prime / E)
        c2 = 1.5 * Om / a ** 2
        return [delta_prime, c1 * delta_prime + c2 * delta]

    res = sp.integrate.solve_ivp(lin_mdp, (a_ini, a_col), [delta_ini, delta_prime_ini],
                                method='Radau', t_eval=[a_eval], atol=1e-8, rtol=1e-8)
    return res['y'][0, 0]


##############################################################
# -------------------- NUMERICAL ANALYSIS --------------------
##############################################################
a_vals = np.logspace(np.log10(a_ini), np.log10(a_col), num=500, endpoint=True)

lin_EdS = np.array([lin_density_perturbation_EdS(a_i,
                                                 delta_ini=delta_ini_EdS,
                                                 delta_prime_ini=delta_prime_ini_EdS) for a_i in a_vals])

non_lin_EdS = np.array([non_lin_density_perturbation_EdS(a_i,
                                                         delta_ini=delta_ini_EdS,
                                                         delta_prime_ini=delta_prime_ini_EdS) for a_i in a_vals])

lin_LCDM = np.array([lin_density_perturbation_LCDM(a_i,
                                                   Om0_lcdm,
                                                   delta_ini=delta_ini_LCDM,
                                                   delta_prime_ini=delta_prime_ini_LCDM) for a_i in a_vals])

non_lin_LCDM = np.array([non_lin_density_perturbation_LCDM(a_i,
                                                           Om0_lcdm,
                                                           delta_ini=delta_ini_LCDM,
                                                           delta_prime_ini=delta_prime_ini_LCDM) for a_i in a_vals])

lin_LsCDM = np.array([lin_density_perturbation_LsCDM(a_i,
                                                     Om0_lscdm,
                                                     delta_ini=delta_ini_LsCDM,
                                                     delta_prime_ini=delta_prime_ini_LsCDM) for a_i in a_vals])

non_lin_LsCDM = np.array([non_lin_density_perturbation_LsCDM(a_i,
                                                             Om0_lscdm,
                                                             delta_ini=delta_ini_LsCDM,
                                                             delta_prime_ini=delta_prime_ini_LsCDM) for a_i in a_vals])


####################################################
# -------------------- PLOTTING --------------------
####################################################
# https://github.com/garrettj403/SciencePlots
import scienceplots
plt.style.use(['science', 'high-vis'])

params = {
    'legend.fontsize': '42.5',
    'axes.labelsize':  '50',
    'figure.figsize':  (18, 12),
    'xtick.labelsize': '50',
    'ytick.labelsize': '50',
    'font.family': 'serif',
    'axes.linewidth':  '3',
}
pylab.rcParams.update(params)
fig, ax0 = plt.subplots()

ax0.plot(a_vals, lin_EdS, color='#DDAA33',
         lw=6.0, label=r'$\delta_{\rm lin,EdS}$')
ax0.plot(a_vals, non_lin_EdS, color='#DDAA33',
         lw=6.0, label=r'$\delta_{\rm non-lin,EdS}$')
ax0.plot(a_vals, lin_LCDM, color='#BB5566',
         lw=6.0, label=r'$\delta_{{\rm lin},\Lambda}$')
ax0.plot(a_vals, non_lin_LCDM, color='#BB5566',
         lw=6.0, label=r'$\delta_{{\rm non-lin},\Lambda}$')
ax0.plot(a_vals, lin_LsCDM, color='#004488',
         lw=6.0, alpha=0.55, label=r'$\delta_{{\rm lin},\Lambda_{\rm s}}$')
ax0.plot(a_vals, non_lin_LsCDM, color='#004488',
         lw=6.0, alpha=0.55, label=r'$\delta_{{\rm non-lin},\Lambda_{\rm s}}$')
ax0.axhline(lin_EdS[-1], color='#000000', ls='-',
            lw=5.0, label=r'$\delta_{\rm c, EdS}$')
ax0.axhline(non_lin_EdS[-1], color='#66CCEE', ls='-',
            lw=5.0, label=r'$\delta_{\infty}$')

ax0.set_xlim(a_ini, 1)
ax0.set_ylim(1e-3, 1e6)

ax0.set_xlabel(r'$a$')
ax0.set_ylabel(r'$\delta$')

ax0.set_yscale('log')
ax0.set_xscale('log')

ax0.set_xticks([1e-3, 1e-2, 1e-1, 1])
ax0.set_yticks([1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
ax0.xaxis.set_ticks_position('both')
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.xaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.yaxis.set_tick_params(which='major', width=2.5, size=13.0, direction='in')
ax0.yaxis.set_tick_params(which='minor', width=2.0, size=6.50, direction='in')
ax0.minorticks_on()

plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig(r'log\figure_1.pdf', format='pdf', dpi=2400)
plt.show()
