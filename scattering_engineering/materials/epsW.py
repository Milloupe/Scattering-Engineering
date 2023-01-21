"""
epsW(lambd, modele)
permittivity of Tungsten

lambd units must be in meters

modele='ordal' default value, Drude model from Ordal  
Valid from 500 nm to 12.3 um
Ordal et al., Appl. Opt. 24, 4493 (1985)

modele='BB', Brendel-Bormann model from Rakic et al
valid from 0.1 to 5 eV
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""

import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsW(lambd, modele='Ordal'):
    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=5.17*1e4
        omega_tau=4.87*1e2
        epsilon = scattering_engineering.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    elif modele=='BB':
        # Parametres exprimes en eV
        f0=0.197
        Gamma_0=0.057
        omega_p=13.22
        sigma_j=np.array((3.754,0.059,0.273,1.912))
        Gamma_j=np.array((3.689,0.277,1.433,4.555))
        omega_j=np.array((0.481,0.985,1.962,5.442))
        f_j=np.array((0.006,0.022,0.136,2.648))
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    return epsilon