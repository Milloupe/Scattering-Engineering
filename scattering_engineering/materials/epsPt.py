"""
epsPt(lambd, modele)
permittivity of Platinium

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
def epsPt(lambd, modele='Ordal'):
    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=4.15*1e4
        omega_tau=5.58*1e2
        epsilon = scattering_engineering.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    elif modele=='BB':
        # Parametres exprimes en eV
        f0=0.333
        Gamma_0=0.08
        omega_p=9.59
        sigma_j=np.array((0.031,0.096,0.766,1.146))
        Gamma_j=np.array((0.498,1.851,2.604,2.891))
        omega_j=np.array((0.782,1.317,3.189,8.236))
        f_j=np.array((0.186,0.665,0.551,2.214))
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    return epsilon