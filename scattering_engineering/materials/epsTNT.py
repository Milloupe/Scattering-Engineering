"""
epsTNT(lambd, modele)
permittivity of TNT

lambd units must be in meters

modele='Todd' default value, Lorentz model from Too et al  
valid from 6 nm to 8 um
Todd et al., Appl. Phys. B 75, 367 (2002)


"""

import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsTNT(lambd, modele='Todd'):
    if modele=='Todd':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        Ap=np.array((6*1e-4,0.4*1e-4,2.5*1e-3,1e-4))
        omega_p=np.array((1349.3,1406.7,1562,1612))
        Gamma_p= np.array((17.5,10,15,10))
        epsilon = scattering_engineering.materials.Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf)
    
    return epsilon