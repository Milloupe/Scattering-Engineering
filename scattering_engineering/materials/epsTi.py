"""
epsTi(lambd, modele)
permittivity of Titanium

lambd units must be in meters

modele='ordal' default value, Drude model from Ordal  
Valid from 500 nm to 12.3 um
Ordal et al., Appl. Opt. 24, 4493 (1985)

modele='BB', Brendel-Bormann model from Rakic et al
valid from 0.05 to 5 eV
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""

import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsTi(lambd, modele='Ordal'):
    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=2.03*1e4
        omega_tau=3.82*1e2
        epsilon = scattering_engineering.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    elif modele=='BB':
        # Parametres exprimes en eV
        f0=0.126
        Gamma_0=0.067
        omega_p=7.29
        sigma_j=np.array((0.463,0.506,0.799,2.854))
        Gamma_j=np.array((1.877,0.1,0.615,4.109))
        omega_j=np.array((1.459,2.661,0.805,19.86))
        f_j=np.array((0.427,0.218,0.513,0.0002))
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    return epsilon