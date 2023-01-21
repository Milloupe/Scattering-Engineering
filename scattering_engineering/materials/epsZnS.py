"""
epsZnS(lambd, modele)
permittivity of ZnS

lambd units must be in meters


modele='Klein', default value modified Sellmeier model
Valid from 0.4 to 13 um
C. A. Klein. Room-temperature dispersion equations for cubic zinc sulfide, Appl. Opt. 25, 1873-1875 (1986)
"""

import scattering_engineering
import scattering_engineering.materials
import numpy as np
def epsZnS(lambd, modele='Klein'):
    lambd=lambd*1e6
    epsilon = 8.393 + (0.14383/(lambd**2 - 0.2421**2)) + (4430.99/(lambd**2 - 36.71**2))
    
        
    return epsilon