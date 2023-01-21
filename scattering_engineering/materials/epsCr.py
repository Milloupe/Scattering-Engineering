"""
epsCr(lambd, modele)
permittivity of Chromium

lambd units must be in meters

modele='BB',  default value, Brendel-Bormann model from Rakic et al
valid from 200nm to 50 um
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""
import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsCr(lambd, modele='BB'):

    if modele=='BB':
        # Parametres exprimes en eV
        f0=0.154
        Gamma_0=0.048
        omega_p=10.75
        sigma_j=np.array((0.115,0.252,0.225,4.903))
        Gamma_j=np.array((4.256,3.957,2.218,6.983))
        omega_j=np.array((0.281,0.584,1.919,6.997))
        f_j=np.array((0.338,0.261,0.817,0.105))
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    
    return epsilon