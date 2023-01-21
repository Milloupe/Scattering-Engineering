"""
epsBe(lambd, modele)
permittivity of Beryllium

lambd units must be in meters

modele='BB',  default value, Brendel-Bormann model from Rakic et al
valid from 200nm to 50 um
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""
import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsBe(lambd, modele='BB'):

    if modele=='BB':
        # Parametres exprimes en eV
        f0=0.081
        Gamma_0=0.035
        omega_p=18.51
        sigma_j=np.array((0.277,3.167,1.446,0.893))
        Gamma_j=np.array((2.956,3.962,2.398,3.904))
        omega_j=np.array((0.131,0.469,2.827,4.318))
        f_j=np.array((0.066,0.067,0.346,0.311))
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    
    return epsilon