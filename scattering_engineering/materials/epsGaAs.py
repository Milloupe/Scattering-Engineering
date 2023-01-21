"""
modele=SV default
permittivity of GaAs in the THz 30-40um
Lorentzian model fitting Palik data
S. Vassant et al., Appl. Phys. Lett. 97, 161101 (2010)
"""

def epsGaAs(lambd,modele='SV'):
#==============================================================================
#     % Function epsilon=epsGaAs(lambd)
#     % Constante GaAs dans le THz 30-40um
#     % lambda est defini en metres
#     % lambda can be a vector of wavelength and then the corresponding
#     % permittivity vector is returned.
#==============================================================================
    # Expression en cm-1 
    if modele == 'SV':
        eps_inf=11
        omega_LO=292.1
        omega_TO=267.98
        Gamma=2.4
        epsilon = eps_inf*(1+(omega_LO**2-omega_TO**2)/(omega_TO**2-(1e-2/lambd)**2-1j*Gamma*1e-2/lambd))
    elif modele=='SV2':
        import numpy as np
        from math import pi
        eps_inf=11
        omega_LO=292.1
        omega_TO=267.98
        Gamma=2.54
        eta_ph = (omega_LO**2-omega_TO**2)/(omega_TO**2-(1e-2/lambd)**2-1j*Gamma*1e-2/lambd)
        ND=1.47*1e24; # dopage en m-3
        e=1.6*1e-19; 
        eps_0=8.85418782*1e-12
        m_e=9.11*1e-31
        m_etoile=0.067*m_e
#        ND_cm=ND*1e6
        omegap_SI =np.sqrt((ND*e**2)/(eps_0*eps_inf*m_etoile) )
        c=3*1e8
        #lambd_p=2*pi*3*1e8/omega_p
        Gamma_e=69.5

#En cm-1
        omega=(1e-2/lambd) 
        
#        wp^2./(v.^2 + 1i*v.*gammap)
#        c=3*1e8
#        eps0=8.854*1e-12
#        ev=1.60218*1e-19        
# meffective et n3D
#        omegap_SI= np.sqrt((n3D*ev**2)/(eps0*epsinf*meffective))
        omega_p = (omegap_SI*1e-2)/(2*pi*c)
        
        eta_e = omega_p**2/(omega**2+1j*Gamma_e*omega)
        
        epsilon=eps_inf*(1+eta_ph-eta_e)
        
    
    return epsilon