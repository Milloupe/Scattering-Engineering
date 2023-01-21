"""
epsSiO2(lambd, modele)
permittivity of Silicon Dioxide

lambd units must be in meters

modele='BB',  default value, Brendel-Bormann model from Kischkat et al.
sample SO16-200
valid from 1.54 to 14.29 um for a thin film
Kischkat et al., Appl. Opt. 51, 6789 (2012)
"""
import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsSiO2(lambd, modele='BB'):

    if modele == 'BB':
        # Parametres exprimes en cm-1
#    Modele pour l'echantillon de Kischkat SO16-200
        f0=0
        Gamma_0=0
        omega_p=1
        eps_infini=2.09
        
        sigma_j=np.array((78,125,62,180))/(np.sqrt(8*np.log(2)))
        Gamma_j=np.array((0.59,1.23,8.52,54.95))
        omega_j=np.array((1046,1167,1058,798))
        f_j=np.array((579,290,457,406))**2 # defini a partir des mu_pj
   
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,eps_inf=eps_infini,units_model='cm-1')
    elif modele == 'BBSO40':
        # Parametres exprimes en cm-1
#    Modele pour l'echantillon de Kischkat SO40-200
        f0=0
        Gamma_0=0
        omega_p=1
        eps_infini=2.09
        
        sigma_j=np.array((63,122,67,427,85))/(np.sqrt(8*np.log(2)))
        Gamma_j=np.array((15.53,4.43,0.42,54.14,12.94))
        omega_j=np.array((1046,1167,1058,434,316))
        f_j=np.array((544,309,466,427,223))**2 # defini a partir des mu_pj
#        Gamma_j=np.array((1.55,10.57,2.24,54.14,12.94))
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,eps_inf=eps_infini,units_model='cm-1')
    elif modele == 'C2N':
        # Parametres exprimes en cm-1
#    Modele pour l'echantillon du C2N Imhotep 
        f0=0
        Gamma_0=0
        omega_p=1
        eps_infini=2.16908
        
        sigma_j=np.array((58.7,70.6,285.1,114.5))/(np.sqrt(8*np.log(2))) #
        Gamma_j=np.array((9.3,15.3,1637,0.0178)) # nu_tj
        omega_j=np.array((954,1053.2,1306.7,1167.5)) # nu_0j 
        f_j=np.array((124.2,721.2,6.6,294))**2 # defini a partir des mu_pj
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,eps_inf=eps_infini,units_model='cm-1')
    elif modele == 'C2N2':
        # Parametres exprimes en cm-1
#    Modele pour l'echantillon du C2N Imhotep 
        f0=0
        Gamma_0=0
        omega_p=1
        eps_infini=2.17
        
        sigma_j=np.array((70.6))/2.3548 #
        Gamma_j=np.array((15.3)) # nu_tj
        omega_j=np.array((1053)) # nu_0j 
        f_j=np.array((721.2))**2 # defini a partir des mu_pj
        epsilon= scattering_engineering.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,eps_inf=eps_infini,units_model='cm-1')
    elif modele == 'const':
        epsilon=1.54**2

    return epsilon