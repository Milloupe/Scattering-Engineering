"""
epsCo(lambd, modele)
permittivity of Cobalt

lambd units must be in meters

modele='ordal' default value, Drude model from Ordal  
valid from 250 nm to 20 um
Ordal et al., Appl. Opt. 24, 4493 (1985)
"""

import scattering_engineering
import scattering_engineering.materials
def epsCo(lambd, modele='Ordal'):
    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=3.2*1e4
        omega_tau=2.95*1e2
        epsilon = scattering_engineering.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    return epsilon