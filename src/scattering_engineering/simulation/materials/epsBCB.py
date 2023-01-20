"""
epsBCB(lambd, modele)
permittivity of BCB

lambd units must be in meters

modele='Chevalier' default value, Lorentz model from These Paul Chevalier et al  


"""

import src.scattering_engineering.simulation
import src.scattering_engineering.simulation.materials
import numpy as np
def epsBCB(lambd, modele='Chevalier'):
    if modele=='Chevalier':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=2.25
        omega_p=np.array((1272,208.4,76.5,159.5,217.8))
        Gamma_p= np.array((27067,60,10.3,110,70.2))
        omega_0=np.array((1024,1045.8,1253.8,2913.3,805.5))
        epsilon = src.scattering_engineering.simulation.materials.Lorentz3_omega(lambd,omega_0,omega_p,Gamma_p,eps_inf)
    
    return epsilon