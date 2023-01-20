#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:39:42 2021

@author: denis
"""

import scattering-engineering.wrapping_model as modele
import numpy as np
import BMM
import BMM.materials as materials
import matplotlib
from joblib import Parallel, delayed

font = {'family' : 'DejaVu',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

### Defining the structure
l_structure = list()
structure = BMM.Bunch()
# The interfaces (note: the given period is supposed to begin with metal
#                       and then alternate between dielectric and metal)
#                (It is also not supposed that there is an interface at x=0 or y=0)
structure.interf = [0.5e-6, 0.75e-6]
structure.prof = [3.e-6]
structure.period = 6e-6
structure.epaiss_metal = structure.prof[0]+1e-6
structure.epaiss_sub = list()

# The materials
structure.eps_1 = 1.0**2 # The permittivity of the semi infinite plane above the surface
structure.eps_2 = [1] # The permittivities in the groove/slits
structure.metal = 'Au'
structure.eps_3 = 1.0  # The permittivity of the anti-reflection coatingr if there is one
structure.eps_sub = list()

l_structure.append(structure)


### Defining the field incidence
l_angle = list()
angle = BMM.Bunch()
angle.theta = 0.1*np.pi/180
# Colatitude of incidence
angle.phi = 0.01*np.pi/180
# The angle between the projection of the incident vector and the x axis (direction of structuration)
angle.psi = 0.01*np.pi/180
# The angle of rotation of the fields around the incident vector. 0 is TM
l_angle.append(angle)
#nb_angle = 6
#max_angle = 60
#for i_angle in range(nb_angle):
#    angle = BMM.Bunch()
#    angle.theta = (0.01 + i_angle*max_angle/(nb_angle-1)) * np.pi/180
#    angle.phi = 0.0*np.pi/180
#    angle.psi = 0.0*np.pi/180
#    l_angle.append(angle) 


### Defining the wavelengths
#l_lambdas = np.arange(4, 17, 0.25)*1e-6
l_lambdas = np.array([4.65])*1e-6
# lambdas will probably always be a simple np.array, no list needed


### Defining the random factors
#l_rf = np.arange(4., 0.4, -0.5)*structure.period
# The meaning of rand_factors depends on the type of randomness chosen
# (e.g. if gaussian, is the std deviation ; if jitter, is the size of the distribution...)
l_rf = np.arange(.01, 1., 0.1)*structure.period
# Specific gaussian noise disorder, second var is variance


### Defining numerical parameters
params = BMM.Bunch()
params.super_period = 500
# The size of the metaperiod taken for the computation
# Larger than 1 is useful only if the structure is disordered
params.nb_modes = 20 * params.super_period
# Up to what number we compute the Rayleigh modes in R = sum[k=-N, N](R_k)
params.nb_reps = 25
# Number of times (with different random draws) the structure's properties
# are computed. Is forced to 1 in rand_factor = 0
params.struct_disorder = "all"
# Which structure to disorder when jittering (only used if type_disorder="jitter" or "geom")
params.unit_lambda_in = "m" # Can be um, Hz or cm-1 (apart from m)
params.unit_lambda_plot = "um" # Can be um, Hz or cm-1 (apart from m)
params.unit_rf_in = "m" # Can be um, Hz or cm-1 (apart from m)
params.unit_rf_plot = "um" # Can be um, Hz or cm-1 (apart from m)
params.unit_struct_in = "m"
params.unit_struct_plot = "um"
params.unit_angle_in = "rad"
params.unit_angle_plot = "deg"
# The units in which variables are given, to convert to more readable types

### TODO ###
"""
- Ajouter autres graphs

Si le temps
- Sauvegarde des fichiers :
    - Rajouter sécurité (pour pas perdre tout le temps de calcul si pas bon dossier)
    - Rajouter perms et epaisseurs substrats dans nom fichier
    - Ajouter nom du type de graph (mais pas d'overwrite parce que pas mêmes variables)
- interfacer BMM
- Bunch intelligent
- Possibilité de mettre plusieurs répétitions sur le même plot (quand besoin)
- Possibilité de choisir l'échelle/la colormap pour les plots 2D
- Eviter de faire encore et encore le test "... in kept_modes"
"""


type_disorder = "Correl"
variables = [l_structure,
            l_angle,
            l_lambdas,
            l_rf]
type_plot = "0DTheta"
# Which plot we want (1DLambda/Theta/RF, 2DLambdaTheta/LambdaRF/LambdaStruct/ThetaStruct)
kept_modes = "refl"
# Which modes to keep, in key words: refl/tran/spec/diff/scat/abs
# (in no particular order)

#%%

path = "Resultats_UCA"
filename = "Desordre_Correl_full_graph_multi"

var = np.arange(0.0, 0.5, 0.05)
multi = len(var)
def calcul(variables, params, var=0, type_disorder=type_disorder):
    
    res, SI_variables = modele.analytic_model(variables, params, variance=var,
                                              type_disorder=type_disorder, progress=False)
    ordx, resx, tr_ordx, tr_resx = res
    # Saving the figures to path/filename_[structure geometry].svg
    modele.post_processing_analytical(res, type_plot, kept_modes,
                                   SI_variables, params, var, save=True,
                                   path=path, file=filename, averaging=True)

Parallel(n_jobs=multi, verbose=20)(delayed(calcul)(variables, params, var=v, type_disorder=type_disorder) for v in var)

# for var in [.1, .2, .3, .4]:
#     res, SI_variables = modele.analytic_model(variables, params, variance=var,
#                                               type_disorder=type_disorder)
#     ordx, resx, tr_ordx, tr_resx = res
#     # Computing the reflected and transmitted orders (res), as well as their positions (ords)
    
    
#     # Saving the figures to path/filename_[structure geometry].svg
#     modele.post_processing_analytical(res, type_plot, kept_modes,
#                                    SI_variables, params, var, save=True,
#                                    path=path, file=filename, averaging=True)
