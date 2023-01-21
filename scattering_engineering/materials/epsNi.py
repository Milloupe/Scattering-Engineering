"""
epsNi(lambd, modele)
permittivity of Nickel

lambd units must be in meters

modele='ordal' default value, Drude model from Ordal  
Valid from 500 nm to 56 um
Ordal et al., Appl. Opt. 24, 4493 (1985)

modele='BB', Brendel-Bormann model from Rakic et al
valid from 0.2 to 5 eV
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""

import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsNi(lambd, modele='Ordal'):
    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=3.94*1e4
        omega_tau=3.52*1e2
        epsilon = scattering_engineering.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    elif modele=='BB':
        # Parametres exprimes en eV
        f0=0.083
        Gamma_0=0.022
        omega_p=15.92
        sigma_j=np.array((0.606,1.454,0.379,0.51))
        Gamma_j=np.array((2.82,0.12,1.822,6.637))
        omega_j=np.array((0.317,1.059,4.583,6.637))
        f_j=np.array((0.357,0.039,0.127,0.654))
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    return epsilon