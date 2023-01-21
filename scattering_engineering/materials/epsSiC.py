"""
epsSiC(lambd, modele)
permittivity of Silicon Carbide SiC

lambd units must be in meters

modele='FM', default value, Lorentz model from F. Marquier thesis
valid from 2 to 20 um

modele='BB' or 'BB_recuit'
lambd units must be in meters
valid from 2 to 8 um for a thin film ~350nm
Measured in 2017

"""

import scattering_engineering
import scattering_engineering.materials
import numpy as np

def epsSiC(lambd,modele='FM'):
    if modele == 'FM':    
    #Parametres en cm-1
        eps_inf=6.7
        omega_LO=969
        omega_TO=793
        Gamma=4.76
        epsilon = eps_inf*(1+(omega_LO**2-omega_TO**2)/(omega_TO**2-(1e-2/lambd)**2-1j*Gamma*1e-2/lambd))    


    if modele=='BB':
        # Parametres exprimes en cm-1
        f0 = 0
        Gamma_0 = 0
        omega_p = 1
        eps_infini = 7.2464
        sigma_j = np.array((104.763,  147.071))
        Gamma_j = np.array((510.7967, 508.9926))
        omega_j = np.array((909.046,  6042.50))
        f_j     = np.array((1263.41,  1069.06))**2 # defini a partir des mu_pj

   
        epsilon = scattering_engineering.materials.Brendel_model(lambd, f0, Gamma_0, omega_p, 
                                             sigma_j, Gamma_j, omega_j, f_j, 
                                             eps_inf=eps_infini, units_model='cm-1')
    
    if modele=='BB_recuit':
        # Parametres exprimes en cm-1
        f0 = 0
        Gamma_0 = 0
        omega_p = 1
        eps_infini = 6.74961
        sigma_j = np.array((14.376,   171.873))
        Gamma_j = np.array((489.8768, 498.5295))
        omega_j = np.array((862.632,  5969.447))
        f_j     = np.array((1208.27,   851.33))**2 # defini a partir des mu_pj

   
        epsilon = scattering_engineering.materials.Brendel_model(lambd, f0, Gamma_0, omega_p, 
                                             sigma_j, Gamma_j, omega_j, f_j, 
                                             eps_inf=eps_infini, units_model='cm-1')
    
    return epsilon
    
