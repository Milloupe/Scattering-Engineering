"""
epsPb(lambd, modele)
permittivity of Lead

lambd units must be in meters

modele='ordal' default value, Drude model from Ordal  
Valid from 500 nm to 56 um
Ordal et al., Appl. Opt. 24, 4493 (1985)
"""

import src.scattering_engineering.simulation
import src.scattering_engineering.simulation.materials
def epsPb(lambd, modele='Ordal'):
    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=5.94*1e4
        omega_tau=16.3*1e2
        epsilon = src.scattering_engineering.simulation.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    return epsilon