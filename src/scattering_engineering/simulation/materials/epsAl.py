"""
epsAl(lambd, modele)
permittivity of Aluminium

lambd units must be in meters

modele='Ordal' default value, Drude model from Ordal  
valid from 500 nm to 56 um
Ordal et al., Appl. Opt. 24, 4493 (1985)

modele='BB', Brendel-Bormann model from Rakic et al
valid from 12nm to 200 um
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""

import src.scattering_engineering.simulation
import src.scattering_engineering.simulation.materials
import numpy as np
def epsAl(lambd, modele='Ordal'):
    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=11.9*1e4
        omega_tau=6.6*1e2
        epsilon = src.scattering_engineering.simulation.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    elif modele=='BB':
        # Parametres exprimes en eV
        f0=0.526
        Gamma_0=0.047
        omega_p=14.98
        sigma_j=np.array((0.013,0.042,0.256,1.735))
        Gamma_j=np.array((0.312,0.315,1.587,2.145))
        omega_j=np.array((0.163,1.561,1.827,4.495))
        f_j=np.array((0.213,0.06,0.182,0.014))
        epsilon= src.scattering_engineering.simulation.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    return epsilon