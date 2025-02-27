import numpy as np
import scipy as sp

from models.radiation_calc import Orh2, Oph2


######################################################
# -------------------- PARAMETERS --------------------
######################################################
c = 299792.458   # speed of light [km/s]

# base plikHM (TT+TE+EE+lowl+lowE+lensing)
Obh2 = 0.022383  # physical baryon density parameter
Omh2 = 0.143140  # physical matter density parameter
z_lss = 1089.914 # redshift to the LSS
r_s =  144.394   # comoving sound horizon at the LSS [Mpc]

theta_true = 0.01041085 # acoustic scale
theta_error = 1e-8      # accepted error in theta
h0_prior = [0.4, 1.0]   # h0 prior range


###################################################################
# --------- COMOVING ANGULAR DIAMETER DISTANCE AT THE LSS ---------
###################################################################
def d_A_finder_LsCDM(h0, z_dag):
    """Calculating the comoving angular diameter distance to the LSS (d_A(z_*))"""
    def integrand(z):
        f_lscdm = np.sign(z_dag - z)
        return c / (100*np.sqrt(Omh2*(1 + z)**3 + Orh2*(1 + z)**4 + (h0**2 - Omh2 - Orh2)*f_lscdm))

    d_A = sp.integrate.quad(integrand, 0, z_lss)[0]
    return d_A


# ########################################################
# --------- HUBBLE CONSTANT & DENSITY PARAMETERS ---------
# ########################################################
def h0_finder_LsCDM(z_dag):
    """Finding the Hubble constant"""
    h0_min, h0_max = h0_prior
    while True:
        h0_test = (h0_min + h0_max) / 2
        d_A_test = d_A_finder_LsCDM(h0_test, z_dag)
        theta_test = r_s / d_A_test

        if abs(theta_true - theta_test) <= theta_error:
            break
        elif theta_true - theta_test > 0:
            h0_min = h0_test
        else:
            h0_max = h0_test

    return h0_test


def Om0_finder_LsCDM(z_dag):
    """Finding the matter density parameter"""
    h0 = h0_finder_LsCDM(z_dag)
    return Omh2 / h0 ** 2


def Ob0_finder_LsCDM(z_dag):
    """Finding the baryon density parameter"""
    h0 = h0_finder_LsCDM(z_dag)
    return Obh2 / h0 ** 2


def Or0_finder_LsCDM(z_dag):
    """Finding the radiation density parameter"""
    h0 = h0_finder_LsCDM(z_dag)
    return Orh2 / h0 ** 2
