#===================================
#Modeles de DNT
#===================================
"""

Modele publi : Density functional theory treatment of the structures and vibrational frequencies of 2,4- and 2,6- dinitrotoluene
Modifie pour un meilleur accord aux experiences

Experience : c=1.72mg/mL, Vdep=5uL en 10  gouttes de 0.5uL _ TM
Mesure FTIR TM 
"""
#import scattering_engineering
#import scattering_engineering.materials
#import numpy as np
#def epsTNT(lambd, modele='Todd'):
#    if modele=='Todd':
#        # Lambd en m
#        # Parametres exprimes en cm-1
#        eps_inf=1
#        Ap=np.array((22*1e-5,60*1e-5,19*1e-5,36*1e-5,11*1e-5, 23.5*1e-5, 51*1e-4,15*1e-5,14*1e-5,15*1e-5,13*1e-5,4.5*1e-5,300*1e-5,310*1e-5,50*1e-5,62*1e-5))
#        omega_p=np.array((1037,1067,1135,1151,1205,1268,1350,1383,1397,1440,1453,1480,1524,1538,1603,1610))
#        Gamma_p= np.array((10,10,10,8,11,10,20,12,10,18,14,10,20,20,10,10))
#        epsilon = scattering_engineering.materials.Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf)
#    
#    return epsilon


"""
Nouveau modele

Modification apres les mesures angulaires 

Experience : c=1.41mg/mL, Vdep=20uL en 10 gouttes de 2uL _ TM

Epaisseur de DNT supposee : hDNT=2.8

Mesure du 28 Juin
"""
#import scattering_engineering
#import scattering_engineering.materials
#import numpy as np
#def epsTNT(lambd, modele='Todd'):
#    if modele=='Todd':
#        # Lambd en m
#        # Parametres exprimes en cm-1
#        eps_inf=1
#        Ap=np.array((2.8*1e-4,3.5*1e-4,1.7*1e-4,2.5*1e-4,2.2*1e-4,3.0*1e-4,3.3*1e-3,1.8*1e-4,1.9*1e-4,1.1*1e-4,1.3*1e-4,1.0*1e-4,2.8*1e-3,1.4*1e-3,0.50*1e-3,0.45*1e-3))
#        omega_p=np.array((1037,1067,1135,1151,1205,1268,1346,1383,1397,1440,1455,1474,1522,1538,1603,1610))
#        Gamma_p= np.array((15,10,11,9,20,15,20,15,15,10,10,10,23,22,15,15))
#        epsilon = scattering_engineering.materials.Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf)
#    
#    return epsilon


"""
Nouveau modele

Modification en prenant en compte la polarisation 

Experience : c=1.41mg/mL, vdep = 20uL en 10 gouttes de 2uL _ TE

Epaisseur de DNT supposee : 
    
Mesure du 26 Juillet
"""

import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsDNT(lambd, modele='Todd'):
    if modele=='Todd':
        # Lambd en m
        # Parametres exprimes en cm-1
        eps_inf=1
        Ap=np.array((4.4*1e-4,6.9*1e-4,3.0*1e-4,5.2*1e-4,3.0*1e-4,4.8*1e-4,5.0*1e-3,2.9*1e-4,2.6*1e-4,1.1*1e-4,1.3*1e-4,1.0*1e-4,2.8*1e-3,1.4*1e-3,0.50*1e-3,0.45*1e-3))
        omega_p=np.array((1037,1067,1135,1151,1205,1268,1350,1383,1397,1440,1455,1474,1522,1538,1603,1610))
        Gamma_p= np.array((15,10,11,9,20,15,20,15,15,10,10,10,23,22,15,15))
        epsilon = scattering_engineering.materials.Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf)
    
    return epsilon

"""
Autres modeles
"""
#hDNT=3.2
#hDNT=3.1
#import scattering_engineering
#import scattering_engineering.materials
#import numpy as np
#def epsTNT(lambd, modele='Todd'):
#    if modele=='Todd':
#        # Lambd en m
#        # Parametres exprimes en cm-1
#        eps_inf=1
#        Ap=np.array((2.2*1e-4,3.0*1e-4,1.5*1e-4,2.1*1e-4,2.0*1e-4,2.9*1e-4,3.0*1e-3,1.7*1e-4,1.9*1e-4,1.0*1e-4,1.2*1e-4,0.9*1e-4,2.8*1e-3,1.4*1e-3,0.50*1e-3,0.45*1e-3))
#        omega_p=np.array((1037,1067,1135,1151,1205,1268,1346,1383,1397,1440,1453,1480,1522,1538,1603,1610))
#        Gamma_p= np.array((15,10,11,9,20,15,20,15,15,10,10,10,23,22,15,15))
#        epsilon = scattering_engineering.materials.Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf)
#    
#    return epsilon

#hDNT=2.9
#import scattering_engineering
#import scattering_engineering.materials
#import numpy as np
#def epsTNT(lambd, modele='Todd'):
#    if modele=='Todd':
#        # Lambd en m
#        # Parametres exprimes en cm-1
#        eps_inf=1
#        Ap=np.array((2.3*1e-4,3.2*1e-4,1.55*1e-4,2.3*1e-4,2.0*1e-4,2.9*1e-4,3.2*1e-3,1.7*1e-4,1.9*1e-4,1.0*1e-4,1.2*1e-4,0.9*1e-4,2.8*1e-3,1.4*1e-3,0.50*1e-3,0.45*1e-3))
#        omega_p=np.array((1037,1067,1135,1151,1205,1268,1346,1383,1397,1440,1453,1480,1522,1538,1603,1610))
#        Gamma_p= np.array((15,10,11,9,20,15,20,15,15,10,10,10,23,22,15,15))
#        epsilon = scattering_engineering.materials.Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf)
#    
#    return epsilon

#hDNT=3.0
#import scattering_engineering
#import scattering_engineering.materials
#import numpy as np
#def epsTNT(lambd, modele='Todd'):
#    if modele=='Todd':
#        # Lambd en m
#        # Parametres exprimes en cm-1
#        eps_inf=1
#        Ap=np.array((2.3*1e-4,3.2*1e-4,1.55*1e-4,2.3*1e-4,2.0*1e-4,2.9*1e-4,3.2*1e-3,1.7*1e-4,1.9*1e-4,1.0*1e-4,1.2*1e-4,0.9*1e-4,2.8*1e-3,1.4*1e-3,0.50*1e-3,0.45*1e-3))
#        omega_p=np.array((1037,1067,1135,1151,1205,1268,1346,1383,1397,1440,1453,1480,1522,1538,1603,1610))
#        Gamma_p= np.array((15,10,11,9,20,15,20,15,15,10,10,10,23,22,15,15))
#        epsilon = scattering_engineering.materials.Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf)
#    
#    return epsilon

#hDNT=2.5
#import scattering_engineering
#import scattering_engineering.materials
#import numpy as np
#def epsTNT(lambd, modele='Todd'):
#    if modele=='Todd':
#        # Lambd en m
#        # Parametres exprimes en cm-1
#        eps_inf=1
#        Ap=np.array((3.1*1e-4,4.1*1e-4,1.9*1e-4,2.9*1e-4,2.4*1e-4,3.4*1e-4,3.6*1e-3,2.0*1e-4,2.2*1e-4,1.1*1e-4,1.5*1e-4,0.9*1e-4,2.8*1e-3,1.5*1e-3,0.4*1e-3,0.6*1e-3))
#        omega_p=np.array((1037,1067,1135,1151,1205,1268,1346,1383,1397,1440,1453,1480,1525,1538,1603,1610))
#        Gamma_p= np.array((15,10,11,9,20,15,20,15,15,10,10,10,23,22,15,15))
#        epsilon = scattering_engineering.materials.Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf)
#    
#    return epsilon



##hDNT=2.3
#import scattering_engineering
#import scattering_engineering.materials
#import numpy as np
#def epsTNT(lambd, modele='Todd'):
#    if modele=='Todd':
#        # Lambd en m
#        # Parametres exprimes en cm-1
#        eps_inf=1
#        Ap=np.array((3.3*1e-4,4.9*1e-4,2.1*1e-4,2.9*1e-4,2.2*1e-4,3.0*1e-4,3.8*1e-3,1.8*1e-4,1.9*1e-4,1.2*1e-4,1.4*1e-4,1.0*1e-4,3.2*1e-3,1.7*1e-3,0.58*1e-3,0.58*1e-3))
#        omega_p=np.array((1037,1067,1135,1151,1205,1268,1346,1383,1397,1440,1453,1480,1522,1538,1603,1610))
#        Gamma_p= np.array((15,10,11,9,20,15,20,15,15,10,10,10,23,22,15,15))
#        epsilon = scattering_engineering.materials.Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf)
#    
#    return epsilon