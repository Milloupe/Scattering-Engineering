"""
epsAg(lambd, modele)
permittivity of Silver

lambd units must be in meters

modele='Ordal', Drude model from Ordal  
valid from 500 nm to 20 um
Ordal et al., Appl. Opt. 24, 4493 (1985)

modele='BB', Brendel-Bormann model from Rakic et al
valid from 0.125 to 6eV
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""
import scattering_engineering as sim
import scattering_engineering.materials as mat
import numpy as np
def epsAg(lambd, modele='Ordal'):

    if modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=7.27*1e4
        omega_tau=1.45*1e2
        epsilon = mat.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    elif modele=='BB':
        # Parametres exprimes en eV
        f0=0.821
        Gamma_0=0.049
        omega_p=9.01
        sigma_j=np.array((1.894,0.665,0.189,1.17,0.516))
        Gamma_j=np.array((0.189,0.067,0.019,0.117,0.052))
        omega_j=np.array((2.025,5.185,4.343,9.809,18.56))
        f_j=np.array((0.05,0.133,0.051,0.467,4))
        epsilon= mat.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
        
    return epsilon