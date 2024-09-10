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


path = "EOT"
filename = "Optim_GSG"

def cost_function(X, plot=False):
    d_slit, d_groove_1, d_groove_2, w_slit, w_groove_1, w_groove_2, per = X
        
    ### Defining the structure
    l_structure = list()
    structure = bunch.Bunch()
    space = (per - w_slit - w_groove_1 - w_groove_2)/3
    structure.interf = [0, w_slit, space, space+w_groove_1, 2*space, 2*space+w_groove_2]
    if (2*space+w_groove_2 > per):
        print("overlap", structure.interf, per)
        return 10
    structure.depth = [d_slit, d_groove_1, d_groove_2]
    structure.period = per
    structure.height_metal = structure.depth[0]#+1e-6
    structure.height_sub = list()

    # The materials
    structure.eps_1 = 1.0**2 # The permittivity of the semi infinite plane above the surface
    structure.eps_2 = [1] # The permittivities in the groove/slits
    structure.metal = 'Au'
    structure.eps_3 = 1.0  # The permittivity of the anti-reflection coatingr if there is one
    structure.eps_sub = list()
    l_structure.append(structure)


    ### Defining the field incidence
    l_angle = list()
    angle = bunch.Bunch()
    nb_angle = 10
    max_angle = 89
    for i_angle in range(nb_angle):
        angle = bunch.Bunch()
        angle.theta = (0.01 + i_angle*max_angle/(nb_angle-1)) * np.pi/180
        angle.phi = 0.0*np.pi/180
        angle.psi = 0.0*np.pi/180
        l_angle.append(angle)


    ### Defining the wavelengths
    lam_step = 10
    l_lambdas = np.concatenate([np.linspace(6,8,lam_step), np.linspace(8,10,lam_step), np.linspace(10,12,lam_step)])*1e-6
    i_beg_lam = lam_step
    i_end_lam = 2*lam_step
    #l_lambdas = np.array([4.65])*1e-6
    # lambdas will always be a simple np.array, no list needed


    ### Defining the random factors
    l_rf = np.array([0.*structure.period]) #np.arange(.01, 1., 0.2)*structure.period


    ### Defining numerical parameters
    params = bunch.Bunch()
    params.super_period = 1
    # The size of the metaperiod taken for the computation
    # Larger than 1 is useful only if the structure is disordered
    params.nb_modes = 10 * params.super_period
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

    type_disorder = "Jitter"
    variables = [l_structure,
                l_angle,
                l_lambdas,
                l_rf]
    kept_modes = "tran_spec"
    # Which modes to keep, in key words: refl/tran/spec/diff/scat/abs
    # (in no particular order)

    #%%


    res, SI_variables = modele.analytic_model(variables, params, variance=0,
                                            type_disorder=type_disorder, progress=False)
    if plot:
        modele.post_processing_analytical(res, "2DLambdaTheta", kept_modes,
                                    SI_variables, params, 0, save=True,
                                    path=path, file=filename, averaging=True,
                                    contours=[0.1*i for i in range(1,9)])
    saved_var = modele.mode_selection(res, kept_modes, variables, params, averaging=True)
    vars = modele.load_var(saved_var, kept_modes, averaging=True)
    trans = vars.t_spec_avg[0,0] # rows are angles, columns are lambdas
    maxi_trans = np.sum(trans[:, i_beg_lam:i_end_lam])
    mini_trans = np.sum(trans[:, :i_beg_lam]) + np.sum(trans[:, i_end_lam:])
    cost = (mini_trans - maxi_trans) / len(l_lambdas)
    return cost

X_min = np.array([3e-6, 2e-6, 1e-6, 0e-6, 0e-6, 0e-6, 3e-6])
X_max = np.array([5e-6, 3e-6, 2e-6, 1.5e-6, 1.5e-6, 1e-6, 6e-6])
# d_slit, d_groove_1, d_groove_2, w_slit, w_groove_1, w_groove_2, per = 3.6e-6, 2.1e-6, 1.3e-6, 0.8e-6, 0.8e-6, 0.6e-6, 4e-6
import PyMoosh as PM

best, conv = PM.QNDE(cost_function, budget=4000, X_min=X_min, X_max=X_max, progression=10)
print(best)
print(conv)
cost_function(best, plot=True)
# cost_function([d_slit, d_groove_1, d_groove_2, w_slit, w_groove_1, w_groove_2, per])

    #%%

