import numpy as np


# ---------- Constants ----------
# https://physics.nist.gov/cuu/Constants/Table/allascii.txt
C = 299792458  # speed of light in m/s
G = 6.67430e-11  # gravitational constant in m^3/(kg s^2)
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant in W/(m^2K^4)

# ---------- Mpc to km conversion ----------
# https://ssd.jpl.nasa.gov/astro_par.html
AU_IN_KM = 149597870.700  # astronomical unit [km]
PC_IN_KM = (180 * 60 * 60 * AU_IN_KM) / np.pi  # parsec [km]
MPC_IN_KM = PC_IN_KM * 1e6  # megaparsec [km]

# ---------- CMB temperature ----------
# https://iopscience.iop.org/article/10.1088/0004-637X/707/2/916
T_CMB = 2.72548  # CMB temperature in K
N_eff = 3.046  # effective neutrino number

# ---------- Radiation energy density calculation ----------
u_photon = 4 * SIGMA_SB * T_CMB**4 / C  # Photon energy density in J/m^3

# https://inspirehep.net/literature/1224741
rho_photon = u_photon / C**2  # Photon density in kg/m^3
rho_neutrino = N_eff * (7 / 8) * (4 / 11)**(4 / 3) * rho_photon  # Neutrino density in kg/m^3
rho_rad = rho_photon + rho_neutrino  # Radiation density in kg/m^3

# Physical radiation density parameter
Oph2 = rho_photon * (8 * np.pi * G / 3) * (MPC_IN_KM / 100)**2
Onh2 = rho_neutrino * (8 * np.pi * G / 3) * (MPC_IN_KM / 100)**2
Orh2 = Oph2 + Onh2

# print("Oph2: {}".format(round(Oph2, 11)))
# print("Onh2: {}".format(round(Onh2, 11)))
# print("Orh2: {}".format(round(Orh2, 11)))
