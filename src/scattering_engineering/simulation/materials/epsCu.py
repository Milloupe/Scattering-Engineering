"""
epsCu(lambd, modele)
permittivity of Copper

lambd units must be in meters

modele='ordal' default value, Drude model from Ordal  
valid from 500 nm to 56 um
Ordal et al., Appl. Opt. 24, 4493 (1985)

modele='BB', Brendel-Bormann model from Rakic et al
valid from 0.1 to 6eV
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""

import src.scattering_engineering.simulation
import src.scattering_engineering.simulation.materials
import numpy as np
def epsCu(lambd, modele='Ordal'):
    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=5.96*1e4
        omega_tau=0.732*1e2
        epsilon = src.scattering_engineering.simulation.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    elif modele=='BB':
        # Parametres exprimes en eV
        f0=0.562
        Gamma_0=0.03
        omega_p=10.83
        sigma_j=np.array((0.562,0.469,1.131,1.719))
        Gamma_j=np.array((0.056,0.047,0.113,0.172))
        omega_j=np.array((0.416,2.849,4.819,8.136))
        f_j=np.array((0.076,0.081,0.324,0.726))
        epsilon= src.scattering_engineering.simulation.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    return epsilon