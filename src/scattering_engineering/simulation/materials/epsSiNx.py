"""
epsSiNx(lambd, modele)
permittivity of Silicon nitride

lambd units must be in meters

modele='BB',  default value, Brendel-Bormann model from Kischkat et al.
sample SN40-300
valid from 1.54 to 14.29 um for a thin film
Kischkat et al., Appl. Opt. 51, 6789 (2012)

modele='MHD',  Maxwell-Helmholtz-Drude model from Cataldo et al.
valid from 15cm-1 to 10000 cm-1 for a thin film
Cataldo et al., Opt. Lett. 37, 4200 (2012)
"""
import src.scattering_engineering.simulation
import src.scattering_engineering.simulation.materials
import numpy as np
def epsSiNx(lambd, modele='BB'):

    if modele=='BB':
        # Parametres exprimes en cm-1
        f0=0
        Gamma_0=0
        omega_p=1
        eps_infini=3.55
        sigma_j=np.array((171,163,145,104,588))
        Gamma_j=np.array((12,0,58,44,0))
        omega_j=np.array((826,925,1063,1185,2577))
        f_j=np.array((902,664,379,172,235))**2 # defini a partir des mu_pj

   
        epsilon= src.scattering_engineering.simulation.materials.Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,eps_inf=eps_infini,units_model='cm-1')
    elif modele=='reviewOE61218':
    # Modele de dispersion de Maxwell-Helmholtz-Drude - parametres en THz
    
        omega=2*np.pi*src.scattering_engineering.simulation.conv_lambda_from_m(lambd,'THz')
        eps_infini=4.603+1j*0.012
       
        eps_prime=np.array((7.499,6.761,6.599,5.305,4.681))
        eps_seconde=np.array((0.,0.376,0.004,0.118,0.207))
        delta_eps=np.array((0.738,0.162,1.294,0.624,0.078))+1j*np.array((-0.376,0.372,-0.114,-0.089,0.195))
        # delta_eps est defini comme eps_j-eps_(j+1)
        omega_T=np.array((75.42,82.51,145.07,154.13,187.32))
        Gamma=np.array((36.51,40.44,17.29,21.878,37.37))
        alpha=np.array((0.,0.34,0.0006,0.0002,0.008))
        Gamma_prime=Gamma*np.exp(-alpha*((omega_T**2-omega**2)/(omega*Gamma))**2)
        epsilon = eps_infini+np.sum(delta_eps*omega_T**2/(omega_T**2-omega**2-1j*omega*Gamma_prime))
        
    elif modele=='MHD':
    # Modele de dispersion de Maxwell-Helmholtz-Drude - parametres en THz
    
        omega=src.scattering_engineering.simulation.conv_lambda_from_m(lambd,'THz')
        eps_infini=4.562+1j*0.0124
       
        eps_prime=np.array((7.499,6.761,6.599,5.305,4.681))
        eps_seconde=np.array((0.,0.376,0.004,0.118,0.207))
        delta_eps=np.array((0.828,0.153,1.17,0.829,0.039))+1j*np.array((-0.3759,0.3718,-0.1138,-0.089,0.195))
        # delta_eps est defini comme eps_j-eps_(j+1)
        omega_T=np.array((13.913,15.053,24.521,26.44,31.724))
        Gamma=np.array((5.81,6.436,2.751,3.482,5.948))
        alpha=np.array((0.00001,0.3427,0.0006,0.0002,0.008))
        Gamma_prime=Gamma*np.exp(-alpha*((omega_T**2-omega**2)/(omega*Gamma))**2)
        epsilon = eps_infini+np.sum(delta_eps*omega_T**2/(omega_T**2-omega**2-1j*omega*Gamma_prime))
    return epsilon