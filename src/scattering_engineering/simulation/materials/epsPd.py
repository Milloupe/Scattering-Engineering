"""
epsPd(lambd, modele)
permittivity of Palladium

lambd units must be in meters

modele='ordal' default value, Drude model from Ordal  
Valid from 500 nm to 12.3 um
Ordal et al., Appl. Opt. 24, 4493 (1985)

modele='BB', Brendel-Bormann model from Rakic et al
valid from 0.1 to 5 eV
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""

import src.scattering_engineering.simulation
import src.scattering_engineering.simulation.materials
import numpy as np
def epsPd(lambd, modele='Ordal'):
    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=4.4*1e4
        omega_tau=1.24*1e2
        epsilon = src.scattering_engineering.simulation.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    elif modele=='BB':
        # Parametres exprimes en eV
        f0=0.33
        Gamma_0=0.009
        omega_p=9.72
        sigma_j=np.array((0.694,0.027,1.167,1.331))
        Gamma_j=np.array((2.343,0.497,2.022,0.119))
        omega_j=np.array((0.066,0.502,2.432,5.987))
        f_j=np.array((0.769,0.093,0.309,0.409))
        epsilon= src.scattering_engineering.simulation.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    return epsilon