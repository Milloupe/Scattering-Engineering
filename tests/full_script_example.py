#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 15:39:42 2021

@author: Denis Langevin
"""

from context import wrap as modele
from context import bunch
import numpy as np
import matplotlib

font = {'family' : 'Sans',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

### Defining the structure
l_structure = list()
structure = bunch.Bunch()
# The interfaces (note: the given period is supposed to begin with metal
#                       and then alternate between dielectric and metal)
#                (It is also not supposed that there is an interface at x=0 or y=0)

# Ana_interf_full = [0.20e-6, 1.0e-06, 1.70e-06, 2.50e-06, 3.20e-06, 3.8e-06]
# Ana_prof_full = [3.6e-06, 2.1e-06, 1.3e-06]
# Ana_Lx = 4e-6
structure.interf = [0.20e-6, 1.0e-06, 3.20e-06, 3.8e-06]
structure.depth = [3.6e-06, 1.3e-06]
structure.period = 4e-6
structure.height_metal = structure.depth[0]#+1e-6
structure.height_sub = list()

# The materials
structure.eps_1 = 1.0**2 # The permittivity of the semi infinite plane above the surface
structure.eps_2 = [1] # The permittivities in the groove/slits
structure.metal = 'Au'
structure.eps_3 = 1.0  # The permittivity of the anti-reflection coatingr if there is one
structure.eps_sub = list()
l_structure.append(structure)

# structure = bunch.Bunch()
# The interfaces (note: the given period is supposed to begin with metal
#                       and then alternate between dielectric and metal)
#                (It is also not supposed that there is an interface at x=0 or y=0)
# structure.interf = [0.5e-6, 0.75e-6]
# structure.depth = [3.e-6]
# structure.period = 7e-6
# structure.height_metal = structure.depth[0]+1e-6
# structure.height_sub = list()

# The materials
# structure.eps_1 = 1.0**2 # The permittivity of the semi infinite plane above the surface
# structure.eps_2 = [1] # The permittivities in the groove/slits
# structure.metal = 'Au'
# structure.eps_3 = 1.0  # The permittivity of the anti-reflection coatingr if there is one
# structure.eps_sub = list()

# l_structure.append(structure)


### Defining the field incidence
l_angle = list()
angle = bunch.Bunch()
angle.theta = 0.1*np.pi/180
# Colatitude of incidence
angle.phi = 0.01*np.pi/180
# The angle between the projection of the incident vector and the x axis (direction of structuration)
angle.psi = 0.01*np.pi/180
# The angle of rotation of the fields around the incident vector. 0 is TM
l_angle.append(angle)
# nb_angle = 90
# max_angle = 89
# for i_angle in range(nb_angle):
#     angle = bunch.Bunch()
#     angle.theta = (0.01 + i_angle*max_angle/(nb_angle-1)) * np.pi/180
#     angle.phi = 0.0*np.pi/180
#     angle.psi = 0.0*np.pi/180
#     l_angle.append(angle)


### Defining the wavelengths
l_lambdas = np.arange(6, 13, 0.01)*1e-6
#l_lambdas = np.array([4.65])*1e-6
# lambdas will always be a simple np.array, no list needed


### Defining the random factors
#l_rf = np.arange(4., 0.4, -0.5)*structure.period
# The meaning of rand_factors depends on the type of randomness chosen
# (e.g. if gaussian, is the std deviation ; if jitter, is the size of the distribution...)
l_rf = np.array([0.*structure.period]) #np.arange(.01, 1., 0.2)*structure.period
# Specific gaussian noise disorder, second var is variance


### Defining numerical parameters
params = bunch.Bunch()
params.super_period = 1
# The size of the metaperiod taken for the computation
# Larger than 1 is useful only if the structure is disordered
params.nb_modes = 20 * params.super_period
# Up to what number we compute the Rayleigh modes in R = sum[k=-N, N](R_k)
params.nb_reps = 1
# Number of times (with different random draws) the structure's properties
# are computed. Is forced to 1 in rand_factor = 0
params.struct_disorder = "all"
# Which structure to disorder when jittering (only used if type_disorder="jitter" or "geom")
params.unit_lambda_in = "m" # Can be um, Hz or cm-1 (apart from m)
params.unit_lambda_plot = "um" # Can be um, Hz or cm-1 (apart from m)
params.unit_rf_in = "m" # Can be um, Hz or cm-1 (apart from m)
params.unit_rf_plot = "um" # Can be um, Hz or cm-1 (apart from m)
params.unit_struct_in = "m" # Can be um or m
params.unit_struct_plot = "um" # Can be um or m
params.unit_angle_in = "rad" # Can be rad or deg
params.unit_angle_plot = "deg" # Can be rad or deg
# The units in which variables are given, to convert to more readable types

### TODO ###
"""
- Ajouter autres graphs

Si le temps
- Sauvegarde des fichiers :
    - Ajouter nom du type de graph (mais pas d'overwrite parce que pas mêmes variables)
- Possibilité de mettre plusieurs répétitions sur le même plot (quand besoin)
- Possibilité de choisir l'échelle/la colormap pour les plots 2D
"""


type_disorder = "Jitter"
variables = [l_structure,
            l_angle,
            l_lambdas,
            l_rf]
type_plot = "1DLambda"
# Which plot we want (1DLambda/Theta/RF/ThetaOut, 2DLambdaTheta/LambdaRF/LambdaStruct/ThetaStruct)
kept_modes = "tran_spec"
# Which modes to keep, in key words: refl/tran/spec/diff/scat/abs
# (in no particular order)

#%%

path = "EOT"
filename = "Fente_sillon_2_spectre"

def compute(variables, params, var=0, type_disorder=type_disorder):

    res, SI_variables = modele.analytic_model(variables, params, variance=var,
                                              type_disorder=type_disorder, progress=True)
    ordx, resx, tr_ordx, tr_resx = res
    # Saving the figures to path/filename_[structure geometry].svg
    modele.post_processing_analytical(res, type_plot, kept_modes,
                                   SI_variables, params, var, save=True,
                                   path=path, file=filename, averaging=True,
                                   contours=[0.1*i for i in range(1,9)])

compute(variables, params, type_disorder=type_disorder)

#%%

