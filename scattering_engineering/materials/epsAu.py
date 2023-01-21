"""
epsAu(lambd, modele)
permittivity of Gold

lambd units must be in meters

modele='Palik', default value Drude model from fitting Palik data
valid from 2 um to 12 um
Bouchon et al., Appl. Phys. Lett. 98, 191109 (2011)


modele='Ordal', Drude model from Ordal  
valid from 250 nm to 20 um
Ordal et al., Appl. Opt. 24, 4493 (1985)

modele='BB', Brendel-Bormann model from Rakic et al
valid from 0.1 to 6eV
Rakic et al., Appl. Opt. 37, 5271 (1998)
"""
import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsAu(lambd, modele='Palik'):

    if modele=='Palik':
        s_Constants_c = 299792458.0;
        s_Constants_e_over_h = 2.417989454e14;
        s_Constants_Hz_of_eV = s_Constants_e_over_h;
        s_Constants_m_of_eV = s_Constants_c/s_Constants_Hz_of_eV;
        lambdap = s_Constants_m_of_eV / 7.8;
        gaga = 0.0075
        epsilon = 1 - 1.0 / ((lambdap/lambd)*(lambdap/lambd + 1j*gaga))
    elif modele=='Palik2':
        lambdap = 159*1e-9;
        gaga = 0.0055
        epsilon = 1 - 1.0 / ((lambdap/lambd)*(lambdap/lambd + 1j*gaga))
    elif modele=='Palik48':
        lambdap = 159*1e-9;
        gaga = 0.0048
        epsilon = 1 - 1.0 / ((lambdap/lambd)*(lambdap/lambd + 1j*gaga))
    elif modele=='Palik048':
        lambdap = 159*1e-9;
        gaga2 = 0.0048
        gaga=gaga2/7
        epsilon = 1 - 1.0 / ((lambdap/lambd)*(lambdap/lambd + 1j*gaga))
    elif modele=='Palik480':
        lambdap = 159*1e-9;
        gaga2 = 0.0048
        gaga=gaga2*7
        epsilon = 1 - 1.0 / ((lambdap/lambd)*(lambdap/lambd + 1j*gaga))
    elif modele=='NL':
#        Modele non local decrit dans Ciraci, Pendry , Chem Phys Chem 2013
#    Parametre pris de Antoine Moreau, arxiv 2017
#     valide seulement en incidence normale
        lambdap = 159*1e-9;
        gaga = 0.0048
        c_lum = 299792458.0
        beta=1.35*1e6/c_lum
        epsilon = 1 - 1.0 / ((lambdap/lambd)*((lambdap/lambd) * (1-beta**2) + 1j*gaga))        
    elif modele=='Ordal':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        omega_p=7.28*1e4
        omega_tau=2.15*1e2
        epsilon = scattering_engineering.materials.Drude_omega(lambd,omega_p,omega_tau,eps_inf)
    elif modele=='BB':
        # Parametres exprimes en eV
        f0=0.77
        Gamma_0=0.05
        omega_p=9.03
        sigma_j=np.array((0.742,0.349,0.83,1.246,1.795))
        Gamma_j=np.array((0.074,0.035,0.083,0.125,0.179))
        omega_j=np.array((0.218,2.885,4.069,6.137,27.97))
        f_j=np.array((0.054,0.050,0.312,0.719,1.648))
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,units_model='eV')
    
    return epsilon