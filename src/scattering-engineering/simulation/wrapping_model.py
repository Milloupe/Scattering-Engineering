#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:03:01 2021

@author: denis
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)

import model as M1fentes
import BMM
from BMM import materials
import desordre.detection_pics as post

def analytic_model(variables, params, return_last_profil=False,
                   type_disorder="None", variance=0, compute_phase=False, progress=True):
    """
        Wrapper function
        variables contains the following information:
        > l_structure contains geometrical and material information
        > l_angle contains the angular incidence information
        > l_lambdas contains the wavelength at which we want to compute
        > l_rf contains the random factors at which we want to compute
        > l_params contains other numerical parameters
            (nb of repetitions, nb of modes and super_period)
        > type_disorder is the type of disorder wanted. Possibilities:
            - "None" -> not disordered
            - "Jitter" -> jitter based disorder, rf = width of the jitter
                          needs struct_disorder to be defined in l_params
            - "Gaussian" -> gaussian distribution of the structure positions
                            rf = standard deviation
            - "Geom" -> geometry disorder, applying jitter to only one interface
                        of each structure. rf = width of the jitter
                        needs struct_disorder to be defined in l_params
            - "LDSeq" -> Low Discrepancy Sequence, deterministic positions
                         depending on a base and increment. rf = base,
                         The increment used is the one giving the lowest discrepancy:
                             (sqrt(5)-1) / 2
            - "RSA" -> Random Sequential Adsorption, or taking the positions
                       completely at random but still avoid overlaps
    """
    ordx = dict()
    resx = dict()
    tr_ordx = dict()
    tr_resx = dict()
    if (compute_phase):
        phase = dict()

    SI_variables = conversion_to_SI(variables, params)
    l_structure, l_angle, l_lambdas, l_rf = SI_variables

    profil = BMM.Bunch()

    profil.nb_reps = params.nb_reps
    profil.super_period = params.super_period
    profil.nb_modes = params.nb_modes
    
    if(not(return_last_profil) and progress):
        nb_tot = len(l_structure) * len(l_angle) * len(l_lambdas) * len(l_rf) * params.nb_reps
        iter = 0
        printProgressBar(0, nb_tot,
                            prefix = 'Progress:',
                            suffix = 'Complete',
                            length=50)
    
    for i_struc in range(len(l_structure)):
        profil.interf = l_structure[i_struc].interf
        profil.prof = l_structure[i_struc].prof
        profil.period = l_structure[i_struc].period
        profil.ep_metal = l_structure[i_struc].epaiss_metal
        profil.ep_sub = l_structure[i_struc].epaiss_sub
        profil.eps_1 = l_structure[i_struc].eps_1
        profil.eps_2 = l_structure[i_struc].eps_2
        profil.metal = l_structure[i_struc].metal
        profil.eps_sub = l_structure[i_struc].eps_sub
        profil.eps_3 = l_structure[i_struc].eps_3


        for i_rf in range(len(l_rf)):
            profil.random_factor = l_rf[i_rf]
            if (profil.random_factor == "max"):
                profil.random_factor = profil.period
            if (profil.random_factor != 0 or type_disorder == "RSA"):
                nb_rep = profil.nb_reps
            else:
                nb_rep = 1

            for i_rep in range(nb_rep):
                # Initialising the structure
                if (profil.random_factor > 0 and type_disorder == "Jitter"):
                    rt = params.struct_disorder
                    M1fentes.init_super_jitter(profil, rand_type=rt)
                elif (profil.random_factor > 0 and type_disorder == "Gaussian"):
                    M1fentes.init_gaussian_super_random(profil)
                elif (profil.random_factor > 0 and type_disorder == "Geom"):
                    rt = params.struct_disorder
                    M1fentes.init_super_random_geometry(profil, rand_type=rt)
                elif (profil.random_factor > 0 and type_disorder == "LDSeq"):
                    M1fentes.init_super_low_discrep_sequence(profil)
                elif (type_disorder == "RSA"):
                    M1fentes.init_super_rsa(profil)
                elif (type_disorder == "Correl"):
                    M1fentes.init_correl_super_random(profil, variance)
                elif (type_disorder == "Sterl_Correl"):
                    M1fentes.init_sterl_correl_super_random(profil, variance)
                else:
                    M1fentes.init_super(profil)
    
                for i_angle in range(len(l_angle)):
                    profil.theta = l_angle[i_angle].theta
                    profil.phi = l_angle[i_angle].phi
                    profil.psi = l_angle[i_angle].psi
    
                    for i_lambdas in range(len(l_lambdas)):
                        profil.lambd = l_lambdas[i_lambdas]
                        profil.eps_m = materials.epsconst(profil.metal, profil.lambd)

                    
                        solve_analytic(profil)
                        # Where everything is done
                        ordx[i_struc, i_rf, i_angle, i_lambdas, i_rep] = profil.prop_ord1
                        resx[i_struc, i_rf, i_angle, i_lambdas, i_rep] = profil.ref_ord[profil.prop_ord1]
                        tr_ordx[i_struc, i_rf, i_angle, i_lambdas, i_rep] = profil.prop_ord3
                        tr_resx[i_struc, i_rf, i_angle, i_lambdas, i_rep] = profil.trans_ord[profil.prop_ord3]
                        
                        if (compute_phase):
                            phase[i_struc, i_rf, i_angle, i_lambdas, i_rep] = np.real(2*profil.kzd*profil.h)
                        
                        if (not(return_last_profil) and progress):
                            iter += 1
                            if (iter%10 == 0):
                                printProgressBar(iter+1, nb_tot,
                                                prefix = 'Progress:',
                                                suffix = 'Complete',
                                                length=50)
    if (not(return_last_profil) and progress):
        printProgressBar(nb_tot-1, nb_tot,
                        prefix = 'Progress:',
                        suffix = 'Complete',
                        length=50)
    if (return_last_profil):
        return (ordx, resx, tr_ordx, tr_resx), SI_variables, profil
    elif (compute_phase):
        return (ordx, resx, tr_ordx, tr_resx), phase, SI_variables
    return (ordx, resx, tr_ordx, tr_resx), SI_variables


def solve_analytic(profil):
    """
        Master function for the analytical model
    """
    M1fentes.init_base(profil)
    M1fentes.init_structure(profil)
    M1fentes.init_Rayleigh(profil)
    M1fentes.init_variables(profil)
    M1fentes.resolution(profil)
    M1fentes.Rnm(profil)
    M1fentes.reflec_orders(profil)
    M1fentes.Tnm(profil)
    M1fentes.transm_orders(profil)


def post_processing_analytical(res, type_plot, kept_modes,
                               SI_variables, params,  variance=0,
                               averaging=True, save=True, contours=0,
                               path="", file=""):
    """
        Post-processing the results and plotting
        > res is simply the result given by analytical_model(...)
        > type_plot tells the function which variables are going
            be used for the plot
            - 1D plots (only case where we can have more than one mode on the same graph):
                a) Wavelength spectrum --- 1DLambda
                b) Angular spectrum --- 1DTheta
                c) Evolution against random factor --- 1DRF
                d) Angular repartition --- 1DThetaOut TODO
            - 2D plots:
                a) Wavelength/Incidence angle --- 2DLambdaTheta
                b) Wavelength/Random factor --- 2DLambdaRF
                c) Wavelength/Out Angle --- 2DLambdaAngleOut
                d) Wavelength/Structure --- 2DLambdaStruct
                e) Incidence Angle/Structure --- 2DThetaStruct
            - subplots, for all previous possibilities, if there are other
               variables than thoes used in the plot, to have them in the
               same subplot group rather than separate plots --- [replace D->S]
               (NOT YET IMPLEMENTED)
        > kept_modes is a string telling the function which modes are kept between:
            - specular reflection/transmission
            - diffraction above/below the surface
            - scattering above/below the surface
            global idea: "refl_spec" / "trans_spec" / "refl_spec_diff_scat"
    """

    l_structure, l_angle, l_lambdas, l_rf = SI_variables
    if (params.nb_reps == 1):
        # No use averaging on only one iteration
        averaging = False

    plot_variables = conversion_to_plot(SI_variables, params)
    
    if(type_plot[:2]=="1D"):
        saved_var = mode_selection(res, kept_modes, plot_variables, params, averaging)
    elif(type_plot[:2]=="2D"):
        saved_var = mode_selection(res, kept_modes, plot_variables, params, averaging, err=False)
        
        """
            The saved arrays are in the following order, but only with kept_modes
               - spec_ref
               - diff_ref
               - scat_ref
               - spec_tr
               - diff_tr
               - scat_tr
               For each of these reponse types, the following arrays are returned
               - *_reps (all values found, one for each repetition)
                  and if averaging is True
               - *_avg (averaged values)
               - *_min (min values)
               - *_max (max values)
        """
    else:
        # Used for scattering plots
        saved_var = scat_selection(res, kept_modes, plot_variables, params, averaging)

    if (type_plot == "1DLambda"):
        plot_Lambda(saved_var, kept_modes, plot_variables, params,
                      averaging, save, path, file, subplots=False)
    elif (type_plot == "1DTheta"):
        plot_Theta(saved_var, kept_modes, plot_variables, params,
                      averaging, save, path, file, subplots=False)
    elif (type_plot == "1DRF"):
        plot_RF(saved_var, kept_modes, plot_variables, params,
                      averaging, save, path, file, subplots=False)

    elif (type_plot == "2DLambdaTheta" and not(params.unit_angle_plot == "kx")):
        plot_LambdaTheta(saved_var, kept_modes, plot_variables, params,
                      averaging, save, path, file, subplots=False, contours=contours)
    elif (type_plot == "2DLambdaTheta" and params.unit_angle_plot == "kx" and params.unit_lambda_plot == "cm-1"):
        plot_SigmaKx(saved_var, kept_modes, plot_variables, params,
                      averaging, save, path, file, subplots=False)
    elif (type_plot == "2DLambdaRF"):
        plot_LambdaRF(saved_var, kept_modes, plot_variables, params,
                      averaging, save, path, file, subplots=False)
    elif (type_plot == "2DThetaRF"):
        plot_ThetaRF(saved_var, kept_modes, plot_variables, params,
                      averaging, save, path, file, subplots=False)
    elif (type_plot == "2DLambdaStruct"):
        plot_LambdaStruct(saved_var, kept_modes, plot_variables, params,
                      averaging, save, path, file, subplots=False)
    elif (type_plot == "2DThetaStruct"):
        plot_ThetaStruct(saved_var, kept_modes, plot_variables, params,
                      averaging, save, path, file, subplots=False)
    # I'll implement automatic subplots if I have the time
    # But for the moment I'll just save everything as svg and
    # make composite plots via Inkscape
   # if (type_plot == "1SLambda"):
   #     plot_Lambda(saved_var, kept_modes, plot_variables, params,
   #                   averaging, save, path, file, subplots=True)
   # if (type_plot == "1STheta"):
   #     plot_Theta(saved_var, kept_modes, plot_variables, params,
   #                   averaging, save, path, file, subplots=True)
   # if (type_plot == "1SRF"):
   #     plot_RF(saved_var, kept_modes, plot_variables, params,
   #                   averaging, save, path, file, subplots=True)
    else:
#        print("Whatever type of plot you asked for (), it's not implemented yet".format(type_plot))
        print("Defaulting to polar scattering plot")
        plot_Scat(saved_var, kept_modes, plot_variables, params, variance,
                      averaging, save, path, file, subplots=False)


def conversion_to_SI(variables, params):
    """
        Converting variables to more readable formats, using instructions in
        params
    """

    l_structure, l_angle, l_lambdas, l_rf = variables
    # For the moment, we won't convert l_structure values, since we can't plot
    # field maps and therefore don't need strucure dimensions

    # Let's start with angles
    if (params.unit_angle_in == "rad"):
        # Nothing to do!
        conv_l_angle = l_angle
    else:
        conv_l_angle = list()
        for iangle in range(len(l_angle)):
            conv_angle = BMM.Bunch()
            if (params.unit_angle_in == "deg"):
                conv_angle.theta = l_angle[iangle].theta * np.pi/180
                conv_angle.psi = l_angle[iangle].psi * np.pi/180
                conv_angle.phi = l_angle[iangle].phi * np.pi/180
            else:
                print("Only degree (deg) and radiant (rad) angles implemented for the moment")
                conv_angle = l_angle[iangle]
            conv_l_angle.append(conv_angle)

    # Let's continue with the structure
    if (params.unit_struct_in == "m"):
        # Nothing to do!
        conv_l_struc = l_structure
    else:
        conv_l_struc = list()
        for istruc in range(len(l_structure)):
            conv_struc = BMM.Bunch()
            if (params.unit_struct_in == "um"):
            # We know the plot unit is different, and we usually only use
            # deg or rad
                conv_struc.interf = [l_structure[istruc].interf[i] * 1e-6
                                    for i in range(len(l_structure[istruc].interf))]
                conv_struc.prof = [l_structure[istruc].prof[i] * 1e-6
                                    for i in range(len(l_structure[istruc].prof))]
                conv_struc.period = l_structure[istruc].period * 1e-6
                conv_struc.epaiss_metal = l_structure[istruc].epaiss_metal * 1e-6
                conv_struc.epaiss_sub = [l_structure[istruc].epaiss_sub[i] * 1e-6
                                    for i in range(len(l_structure[istruc].epaiss_sub))]
            else:
                print("Only um and m possible as input format for the structure at the moment",
                "no modification done.")
                conv_struc = l_structure[iangle]
            conv_l_struc.append(conv_struc)

    # Let's move to Lambda
    if (params.unit_lambda_in == "m"):
        # Nothing to do!
        conv_l_lambdas = l_lambdas
    else:
        # First convert back to meters
        if (params.unit_lambda_in == "um"):
            conv_l_lambdas = l_lambdas*1e-6
        elif (params.unit_lambda_in == 'cm-1'):
            if 0 not in l_lambdas:
                conv_l_lambdas = 1/(1e2*l_lambdas)
            else:
                print("Lambda=0cm-1 doesn't mean anything, please check your input.")
        elif (params.unit_lambda_in == 'Hz'):
            if 0 not in l_lambdas:
                conv_l_lambdas = 299792458/l_lambdas
            else:
                print("Lambda=0Hz doesn't mean anything, please check your input.")
        else:
            print("Conversion only implemented for um, cm-1 and Hz for the moment.")

    # And now RF (separately from Lambda because sometimes Lambda is given as a
    #             frequency or as a wave number)
    if (params.unit_rf_in == "m"):
        # Nothing to do!
        conv_l_rf = l_rf
    else:
        # First convert back to meters
        if (params.unit_rf_in == "nm"):
            conv_l_rf = l_rf*1e-9
        if (params.unit_rf_in == "um"):
            conv_l_rf = l_rf*1e-6
        elif (params.unit_rf_in == "mm"):
            conv_l_rf = l_rf*1e-3

    SI_variables = conv_l_struc, conv_l_angle, conv_l_lambdas, conv_l_rf
    return SI_variables


def conversion_to_plot(SI_variables, params):
    """
        Converting variables to more readable formats, using instructions in
        params
    """

    l_structure, l_angle, l_lambdas, l_rf = SI_variables
    # For the moment, we won't convert l_structure values, since we can't plot
    # field maps and therefore don't need strucure dimensions

    # Let's start with angles
    if (params.unit_angle_plot == "rad"):
        # Nothing to do!
        conv_l_angle = l_angle
    else:
        conv_l_angle = list()
        for iangle in range(len(l_angle)):
            conv_angle = BMM.Bunch()
            if (params.unit_angle_plot == "deg"):
                conv_angle.theta = l_angle[iangle].theta * 180/np.pi
                conv_angle.psi = l_angle[iangle].psi * 180/np.pi
                conv_angle.phi = l_angle[iangle].phi * 180/np.pi     
            elif (params.unit_angle_plot == "kx"):
                conv_angle.theta = np.sin(l_angle[iangle].theta)
                conv_angle.psi = l_angle[iangle].psi
                conv_angle.phi = l_angle[iangle].phi
            else:
                print("Only degree (deg), radiant (rad) and kx angles implemented for the moment")
                conv_angle = l_angle[iangle]
            conv_l_angle.append(conv_angle)

    # Let's continue with the structure
    if (params.unit_struct_plot == "m"):
        # Nothing to do!
        conv_l_struc = l_structure
    else:
        conv_l_struc = list()
        for istruc in range(len(l_structure)):
            conv_struc = BMM.Bunch()
            if (params.unit_struct_plot == "um"):
            # We know the plot unit is different, and we usually only use
            # deg or rad
                conv_struc.interf = [l_structure[istruc].interf[i] * 1e6
                                    for i in range(len(l_structure[istruc].interf))]
                conv_struc.prof = [l_structure[istruc].prof[i] * 1e6
                                    for i in range(len(l_structure[istruc].prof))]
                conv_struc.period = l_structure[istruc].period * 1e6
                conv_struc.epaiss_metal = l_structure[istruc].epaiss_metal * 1e6
                conv_struc.epaiss_sub = [l_structure[istruc].epaiss_sub[i] * 1e6
                                    for i in range(len(l_structure[istruc].epaiss_sub))]
            else:
                print("Only um and m possible as input format for the structure at the moment",
                "no modification done.")
                conv_struc = l_structure[iangle]
            conv_l_struc.append(conv_struc)

    # Let's move to Lambda
    if (params.unit_lambda_plot == "m"):
        # Nothing to do!
        conv_l_lambdas = l_lambdas
    else:
        # First convert back to meters
        if (params.unit_lambda_plot == "um"):
            conv_l_lambdas = l_lambdas*1e6
        elif (params.unit_lambda_plot == 'cm-1'):
            if 0 not in l_lambdas:
                conv_l_lambdas = 1/(1e2*l_lambdas)
        elif (params.unit_lambda_plot == 'Hz'):
            if 0 not in l_lambdas:
                conv_l_lambdas = 299792458/l_lambdas
        else:
            print("Conversion only implemented for um, cm-1 and Hz for the moment.")

    # And now RF (separately from Lambda because sometimes Lambda is given as a
    #             frequency or as a wave number)
    if (l_rf[0] == "max"):
        conv_l_rf = [0]
    else:
        conv_l_rf = l_rf
        if (params.unit_rf_plot == "m"):
            # Nothing to do!
            conv_l_rf = conv_l_rf
        else:
            # Then convert to the desired unit
            if (params.unit_rf_plot == "nm"):
                conv_l_rf = conv_l_rf*1e9
            if (params.unit_rf_plot == "um"):
                conv_l_rf = conv_l_rf*1e6
            elif (params.unit_rf_plot == "mm"):
                conv_l_rf = conv_l_rf*1e3

    plot_variables = conv_l_struc, conv_l_angle, conv_l_lambdas, conv_l_rf
    return plot_variables


def mode_selection(res, kept_modes, variables, params, averaging=True, err=True):
    """
        Computes the modes wanted for plotting, indicated by kept_modes.
        - averaging tells the function if it should average on repetitions
        -> This is a pretty ugly but quite straightforward function,
           it just has to do very similar things for all cases of kept_modes
        >> Return format is a list of numpy arrays.
           These arrays are in the following order, but only with kept_modes
           - spec_ref
           - diff_ref
           - scat_ref
           - spec_tr
           - diff_tr
           - scat_tr
           For each of these reponse types, the following arrays are returned
           - *_reps (all values found, one for each repetition)
              and if averaging is True
           - *_avg (averaged values)
           - *_min (min values)
           - *_max (max values)
    """

    l_structure, l_angle, l_lambdas, l_rf = variables
    r_ord, r_res, t_ord, t_res = res

    sup = params.super_period

    # Initialising most variables as dict to save space (we won't keep them all)

    if ("refl" in kept_modes):
        r_spec_reps = dict()
        r_diff_reps = dict()
        r_scat_reps = dict()
        if (averaging):
            r_spec_avg = dict()
            r_diff_avg = dict()
            r_scat_avg = dict()
            if (err):
                r_spec_min = dict()
                r_spec_max = dict()
                r_diff_min = dict()
                r_diff_max = dict()
                r_scat_min = dict()
                r_scat_max = dict()
    if ("tran" in kept_modes):
        t_spec_reps = dict()
        t_diff_reps = dict()
        t_scat_reps = dict()
        if (averaging):
            t_spec_avg = dict()
            t_diff_avg = dict()
            t_scat_avg = dict()
            if (err):
                t_spec_min = dict()
                t_spec_max = dict()
                t_diff_min = dict()
                t_diff_max = dict()
                t_scat_min = dict()
                t_scat_max = dict()
    if ("abs" in kept_modes):
        abs_reps = dict()
        t_spec_reps = dict()
        t_diff_reps = dict()
        t_scat_reps = dict()
        r_spec_reps = dict()
        r_diff_reps = dict()
        r_scat_reps = dict()
        if (averaging):
            abs_avg = dict()
            if (err):
                abs_min = dict()
                abs_max = dict()

    # Computing all variables we want to keep and storing them in dict
    for istru in range(len(l_structure)):
        for iangl in range(len(l_angle)):
            for irf in range(len(l_rf)):
                for ilamb in range(len(l_lambdas)):
                    if (l_rf[irf] == 0):
                        n_reps = 1
                    else:
                        n_reps = params.nb_reps

                    if ("refl" in kept_modes or "abs" in kept_modes):
                        r_spec_index = np.where(np.array(r_ord[istru, irf, iangl, ilamb, 0]) == 0)[0]
                        r_spec_reps[istru, irf, iangl, ilamb] = np.concatenate([r_res[istru, irf, iangl, ilamb, l][r_spec_index] for l in range(n_reps)])
                        if (n_reps == 1 and averaging):
                            # Making sure what we have in the end is easily transformed
                            # in a numpy array, i.e. has the same number of
                            # values on all axis.
                            r_spec_reps[istru, irf, iangl, ilamb] = np.array([r_spec_reps[istru, irf, iangl, ilamb] for l in range(params.nb_reps)])

                        if (averaging and "refl" in kept_modes):
                            r_spec_avg[istru, irf, iangl, ilamb] = np.mean(r_spec_reps[istru, irf, iangl, ilamb])
                            if (err):
                                r_spec_min[istru, irf, iangl, ilamb] = np.min(r_spec_reps[istru, irf, iangl, ilamb])
                                r_spec_max[istru, irf, iangl, ilamb] = np.max(r_spec_reps[istru, irf, iangl, ilamb])

                        if ("diff" in kept_modes or "scat" in kept_modes or "abs" in kept_modes):
                            r_diff_index = np.where(np.array(r_ord[istru, irf, iangl, ilamb, 0]) % sup == 0)[0]
                            r_diff_array = np.array([r_res[istru, irf, iangl, ilamb, l][r_diff_index] for l in range(n_reps)])
                            r_diff_reps[istru, irf, iangl, ilamb] = np.sum(r_diff_array, axis=-1) - r_spec_reps[istru, irf, iangl, ilamb]
                            if (n_reps == 1 and averaging):
                                # Making sure what we have in the end is easily transformed
                                # in a numpy array, i.e. has the same number of
                                # values on all axis.
                                r_diff_reps[istru, irf, iangl, ilamb] = np.array([r_diff_reps[istru, irf, iangl, ilamb] for l in range(params.nb_reps)])

                            if (averaging and "refl" in kept_modes):
                                r_diff_avg[istru, irf, iangl, ilamb] = np.mean(r_diff_reps[istru, irf, iangl, ilamb])
                                if (err):
                                    r_diff_min[istru, irf, iangl, ilamb] = np.min(r_diff_reps[istru, irf, iangl, ilamb])
                                    r_diff_max[istru, irf, iangl, ilamb] = np.max(r_diff_reps[istru, irf, iangl, ilamb])

                            if ("scat" in kept_modes or "abs" in kept_modes):
                                r_scat_array = np.array([r_res[istru, irf, iangl, ilamb, l] for l in range(n_reps)])
                                r_scat_reps[istru, irf, iangl, ilamb] = np.sum(r_scat_array, axis=-1) - r_spec_reps[istru, irf, iangl, ilamb] - r_diff_reps[istru, irf, iangl, ilamb]
                                if (n_reps == 1 and averaging):
                                    # Making sure what we have in the end is easily transformed
                                    # in a numpy array, i.e. has the same number of
                                    # values on all axis.
                                    r_scat_reps[istru, irf, iangl, ilamb] = np.array([r_scat_reps[istru, irf, iangl, ilamb] for l in range(params.nb_reps)])

                                if (averaging and "refl" in kept_modes):
                                    r_scat_avg[istru, irf, iangl, ilamb] = np.mean(r_scat_reps[istru, irf, iangl, ilamb])
                                    if (err):
                                        r_scat_min[istru, irf, iangl, ilamb] = np.min(r_scat_reps[istru, irf, iangl, ilamb])
                                        r_scat_max[istru, irf, iangl, ilamb] = np.max(r_scat_reps[istru, irf, iangl, ilamb])


                    if ("tran" in kept_modes or "abs" in kept_modes):
                        t_spec_index = np.where(np.array(t_ord[istru, irf, iangl, ilamb, 0]) == 0)[0]
                        t_spec_reps[istru, irf, iangl, ilamb] = np.concatenate([t_res[istru, irf, iangl, ilamb, l][t_spec_index] for l in range(n_reps)])
                        if (n_reps == 1 and averaging):
                            # Making sure what we have in the end is easily transformed
                            # in a numpy array, i.e. has the same number of
                            # values on all axis.
                            t_spec_reps[istru, irf, iangl, ilamb] = np.array([t_spec_reps[istru, irf, iangl, ilamb] for l in range(params.nb_reps)])

                        if (averaging and "tran" in kept_modes):
                            t_spec_avg[istru, irf, iangl, ilamb] = np.mean(t_spec_reps[istru, irf, iangl, ilamb])
                            if (err):
                                t_spec_min[istru, irf, iangl, ilamb] = np.min(t_spec_reps[istru, irf, iangl, ilamb])
                                t_spec_max[istru, irf, iangl, ilamb] = np.max(t_spec_reps[istru, irf, iangl, ilamb])

                        if ("diff" in kept_modes or "scat" in kept_modes or "abs" in kept_modes):
                            t_diff_index = np.where(np.array(t_ord[istru, irf, iangl, ilamb, 0]) % sup == 0)[0]
                            t_diff_array = np.array([t_res[istru, irf, iangl, ilamb, l][t_diff_index] for l in range(n_reps)])
                            t_diff_reps[istru, irf, iangl, ilamb] = np.sum(t_diff_array, axis=-1) - t_spec_reps[istru, irf, iangl, ilamb]
                            if (n_reps == 1 and averaging):
                                # Making sure what we have in the end is easily transformed
                                # in a numpy array, i.e. has the same number of
                                # values on all axis.
                                t_diff_reps[istru, irf, iangl, ilamb] = np.array([t_diff_reps[istru, irf, iangl, ilamb] for l in range(params.nb_reps)])

                            if (averaging and "tran" in kept_modes):
                                t_diff_avg[istru, irf, iangl, ilamb] = np.mean(t_diff_reps[istru, irf, iangl, ilamb])
                                if (err):
                                    t_diff_min[istru, irf, iangl, ilamb] = np.min(t_diff_reps[istru, irf, iangl, ilamb])
                                    t_diff_max[istru, irf, iangl, ilamb] = np.max(t_diff_reps[istru, irf, iangl, ilamb])

                            if ("scat" in kept_modes or "abs" in kept_modes):
                                t_scat_array = np.array([t_res[istru, irf, iangl, ilamb, l] for l in range(n_reps)])
                                t_scat_reps[istru, irf, iangl, ilamb] = np.sum(t_scat_array, axis=-1) - t_spec_reps[istru, irf, iangl, ilamb] - t_diff_reps[istru, irf, iangl, ilamb]
                                if (n_reps == 1 and averaging):
                                    # Making sure what we have in the end is easily transformed
                                    # in a numpy array, i.e. has the same number of
                                    # values on all axis.
                                    t_scat_reps[istru, irf, iangl, ilamb] = np.array([t_scat_reps[istru, irf, iangl, ilamb] for l in range(params.nb_reps)])

                                if (averaging and "tran" in kept_modes):
                                    t_scat_avg[istru, irf, iangl, ilamb] = np.mean(t_scat_reps[istru, irf, iangl, ilamb])
                                    if (err):
                                        t_scat_min[istru, irf, iangl, ilamb] = np.min(t_scat_reps[istru, irf, iangl, ilamb])
                                        t_scat_max[istru, irf, iangl, ilamb] = np.max(t_scat_reps[istru, irf, iangl, ilamb])

                    if ("abs" in kept_modes):
                        abs_reps[istru, irf, iangl, ilamb] = 1 - np.sum(t_scat_array, axis=-1) - np.sum(r_scat_array, axis=-1)
                        if (n_reps == 1 and averaging):
                            # Making sure what we have in the end is easily transformed
                            # in a numpy array, i.e. has the same number of
                            # values on all axis.
                            abs_reps[istru, irf, iangl, ilamb] = np.array([abs_reps[istru, irf, iangl, ilamb] for l in range(params.nb_reps)])

                        if (averaging):
                            abs_avg[istru, irf, iangl, ilamb] = np.mean(abs_reps[istru, irf, iangl, ilamb])
                            if (err):
                                abs_min[istru, irf, iangl, ilamb] = np.min(abs_reps[istru, irf, iangl, ilamb])
                                abs_max[istru, irf, iangl, ilamb] = np.max(abs_reps[istru, irf, iangl, ilamb])

    # And now saving the kept modes in usable format (numpy arrays)
    # (hopefully, efficiently by list comprehension)
    save = list()
    if ("refl" in kept_modes):
        if("spec" in kept_modes):
            if not(averaging):
                r_spec_reps = np.array([[[[r_spec_reps[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(r_spec_reps)
            if (averaging):
                r_spec_avg = np.array([[[[r_spec_avg[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(r_spec_avg)
                if (err):
                    r_spec_min = np.array([[[[r_spec_min[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    r_spec_max = np.array([[[[r_spec_max[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    save.append(r_spec_min)
                    save.append(r_spec_max)
        if("diff" in kept_modes):
            if not(averaging):
                r_diff_reps = np.array([[[[r_diff_reps[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(r_diff_reps)
            if (averaging):
                r_diff_avg = np.array([[[[r_diff_avg[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(r_diff_avg)
                if (err):
                    r_diff_min = np.array([[[[r_diff_min[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    r_diff_max = np.array([[[[r_diff_max[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    save.append(r_diff_min)
                    save.append(r_diff_max)
        if("scat" in kept_modes):
            if not(averaging):
                r_scat_reps = np.array([[[[r_scat_reps[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(r_scat_reps)
            if (averaging):
                r_scat_avg = np.array([[[[r_scat_avg[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(r_scat_avg)
                if (err):
                    r_scat_min = np.array([[[[r_scat_min[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    r_scat_max = np.array([[[[r_scat_max[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    save.append(r_scat_min)
                    save.append(r_scat_max)

    if ("tran" in kept_modes):
        if("spec" in kept_modes):
            if not(averaging):
                t_spec_reps = np.array([[[[t_spec_reps[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(t_spec_reps)
            if (averaging):
                t_spec_avg = np.array([[[[t_spec_avg[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(t_spec_avg)
                if (err):
                    t_spec_min = np.array([[[[t_spec_min[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    t_spec_max = np.array([[[[t_spec_max[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    save.append(t_spec_min)
                    save.append(t_spec_max)
        if("diff" in kept_modes):
            if not(averaging):
                t_diff_reps = np.array([[[[t_diff_reps[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(t_diff_reps)
            if (averaging):
                t_diff_avg = np.array([[[[t_diff_avg[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(t_diff_avg)
                if (err):
                    t_diff_min = np.array([[[[t_diff_min[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    t_diff_max = np.array([[[[t_diff_max[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    save.append(t_diff_min)
                    save.append(t_diff_max)
        if("scat" in kept_modes):
            if not(averaging):
                t_scat_reps = np.array([[[[t_scat_reps[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(t_scat_reps)
            if (averaging):
                t_scat_avg = np.array([[[[t_scat_avg[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(t_scat_avg)
                if (err):
                    t_scat_min = np.array([[[[t_scat_min[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    t_scat_max = np.array([[[[t_scat_max[istru, irf, iangl, ilamb]
                                            for ilamb in range(len(l_lambdas))]
                                            for iangl in range(len(l_angle))]
                                            for irf in range(len(l_rf))]
                                            for istru in range(len(l_structure))])
                    save.append(t_scat_min)
                    save.append(t_scat_max)

    if("abs" in kept_modes):
        if not(averaging):
            abs_reps = np.array([[[[abs_reps[istru, irf, iangl, ilamb]
                                    for ilamb in range(len(l_lambdas))]
                                    for iangl in range(len(l_angle))]
                                    for irf in range(len(l_rf))]
                                    for istru in range(len(l_structure))])
            save.append(abs_reps)
        if (averaging):
            abs_avg = np.array([[[[abs_avg[istru, irf, iangl, ilamb]
                                    for ilamb in range(len(l_lambdas))]
                                    for iangl in range(len(l_angle))]
                                    for irf in range(len(l_rf))]
                                    for istru in range(len(l_structure))])
            save.append(abs_avg)
            if (err):
                abs_min = np.array([[[[abs_min[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                abs_max = np.array([[[[abs_max[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(abs_min)
                save.append(abs_max)
    return save


def scat_selection(res, kept_modes, variables, params, averaging=True, err=False):
    """
        Computes the modes wanted for plotting, indicated by kept_modes.
        - averaging tells the function if it should average on repetitions
        -> This is a pretty ugly but quite straightforward function,
           it just has to do very similar things for all cases of kept_modes
        >> Return format is a list of numpy arrays.
           These arrays are in the following order, but only with kept_modes
           - spec_ref
           - diff_ref
           - scat_ref
           - spec_tr
           - diff_tr
           - scat_tr
           For each of these reponse types, the following arrays are returned
           - *_reps (all values found, one for each repetition)
              and if averaging is True
           - *_avg (averaged values)
           - *_min (min values)
           - *_max (max values)
    """

    l_structure, l_angle, l_lambdas, l_rf = variables
    r_ord, r_res, t_ord, t_res = res


    # Initialising most variables as dict to save space (we won't keep them all)

    if ("refl" in kept_modes):
        r_out_reps = dict()
        r_out_angl = dict()
        if (averaging):
            r_out_avg = dict()
            if (err):
                r_out_min = dict()
                r_out_max = dict()
    if ("trans" in kept_modes):
        t_out_reps = dict()
        t_out_angl = dict()
        if (averaging):
            t_out_avg = dict()
            if (err):
                t_out_min = dict()
                t_out_max = dict()

    # Computing all variables we want to keep and storing them in dict
    for istru in range(len(l_structure)):
        for iangl in range(len(l_angle)):
            for irf in range(len(l_rf)):
                for ilamb in range(len(l_lambdas)):
                    if (l_rf[irf] == 0):
                        n_reps = 1
                    else:
                        n_reps = params.nb_reps

                    
                    r_out_reps[istru, irf, iangl, ilamb] = np.array([r_res[istru, irf, iangl, ilamb, l] for l in range(n_reps)])
                    r_out_angl[istru, irf, iangl, ilamb] = np.array(r_ord[istru, irf, iangl, ilamb, 0])
                    if (n_reps == 1 and averaging):
                        # Making sure what we have in the end is easily transformed
                        # in a numpy array, i.e. has the same number of
                        # values on all axis.
                        r_out_reps[istru, irf, iangl, ilamb] = np.array([r_out_reps[istru, irf, iangl, ilamb][0] for l in range(params.nb_reps)])
                    if (averaging and "refl" in kept_modes):
                        r_out_avg[istru, irf, iangl, ilamb] = np.mean(r_out_reps[istru, irf, iangl, ilamb], axis=0)
                        if (err):
                            r_out_min[istru, irf, iangl, ilamb] = np.min(r_out_reps[istru, irf, iangl, ilamb], axis=0)
                            r_out_max[istru, irf, iangl, ilamb] = np.max(r_out_reps[istru, irf, iangl, ilamb], axis=0)


                    if ("tran" in kept_modes or "abs" in kept_modes):
                       
                        t_out_reps[istru, irf, iangl, ilamb] = np.array([t_res[istru, irf, iangl, ilamb, l] for l in range(n_reps)])
                        t_out_angl[istru, irf, iangl, ilamb] = np.array(t_ord[istru, irf, iangl, ilamb, 0])
                        if (n_reps == 1 and averaging):
                            # Making sure what we have in the end is easily transformed
                            # in a numpy array, i.e. has the same number of
                            # values on all axis.
                            t_out_reps[istru, irf, iangl, ilamb] = np.array([t_out_reps[istru, irf, iangl, ilamb] for l in range(params.nb_reps)])
    
                        if (averaging and "refl" in kept_modes):
                            t_out_avg[istru, irf, iangl, ilamb] = np.mean(t_out_reps[istru, irf, iangl, ilamb], axis=0)
                            if (err):
                                t_out_min[istru, irf, iangl, ilamb] = np.min(t_out_reps[istru, irf, iangl, ilamb], axis=0)
                                t_out_max[istru, irf, iangl, ilamb] = np.max(t_out_reps[istru, irf, iangl, ilamb], axis=0)

    # And now saving the kept modes in usable format (numpy arrays)
    # (hopefully, efficiently by list comprehension)
    save = list()
    if ("refl" in kept_modes):
        if not(averaging):
            r_out_reps = np.array([[[[r_out_reps[istru, irf, iangl, ilamb]
                                    for ilamb in range(len(l_lambdas))]
                                    for iangl in range(len(l_angle))]
                                    for irf in range(len(l_rf))]
                                    for istru in range(len(l_structure))])
            save.append(r_out_reps)
        if (averaging):
            r_out_avg = np.array([[[[r_out_avg[istru, irf, iangl, ilamb]
                                    for ilamb in range(len(l_lambdas))]
                                    for iangl in range(len(l_angle))]
                                    for irf in range(len(l_rf))]
                                    for istru in range(len(l_structure))])
            save.append(r_out_avg)
            if (err):
                r_out_min = np.array([[[[r_out_min[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                r_out_max = np.array([[[[r_out_max[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(r_out_min)
                save.append(r_out_max)
        r_out_angl = np.array([[[[r_out_angl[istru, irf, iangl, ilamb]
                                for ilamb in range(len(l_lambdas))]
                                for iangl in range(len(l_angle))]
                                for irf in range(len(l_rf))]
                                for istru in range(len(l_structure))])
        save.append(r_out_angl)
        

    if ("tran" in kept_modes):
        if not(averaging):
            t_out_reps = np.array([[[[t_out_reps[istru, irf, iangl, ilamb]
                                    for ilamb in range(len(l_lambdas))]
                                    for iangl in range(len(l_angle))]
                                    for irf in range(len(l_rf))]
                                    for istru in range(len(l_structure))])
            save.append(t_out_reps)
        if (averaging):
            t_out_avg = np.array([[[[t_out_avg[istru, irf, iangl, ilamb]
                                    for ilamb in range(len(l_lambdas))]
                                    for iangl in range(len(l_angle))]
                                    for irf in range(len(l_rf))]
                                    for istru in range(len(l_structure))])
            save.append(t_out_avg)
            if (err):
                t_out_min = np.array([[[[t_out_min[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                t_out_max = np.array([[[[t_out_max[istru, irf, iangl, ilamb]
                                        for ilamb in range(len(l_lambdas))]
                                        for iangl in range(len(l_angle))]
                                        for irf in range(len(l_rf))]
                                        for istru in range(len(l_structure))])
                save.append(t_out_min)
                save.append(t_out_max)
        t_out_angl = np.array([[[[t_out_angl[istru, irf, iangl, ilamb]
                                for ilamb in range(len(l_lambdas))]
                                for iangl in range(len(l_angle))]
                                for irf in range(len(l_rf))]
                                for istru in range(len(l_structure))])
        save.append(t_out_angl)
  
    return save


def load_var(saved_var, kept_modes, averaging, err=True):
    """
        Load the necessary variables for plotting
        - err tells the function whether or not to load the error bars
    """

    var = BMM.Bunch()
    i_mode = 0
    # This will serve as an index in the saved_var list
    # that will increase depending on kept_modes
    if ("refl" in kept_modes):
        if ("spec" in kept_modes):
            if not(averaging):
                var.r_spec_reps = saved_var[i_mode]
                i_mode += 1
            if (averaging):
                var.r_spec_avg = saved_var[i_mode]
                i_mode += 1
                if (err):
                    r_spec_min = saved_var[i_mode]
                    r_spec_max = saved_var[i_mode+1]
                    var.err_r_spec_above = r_spec_max - var.r_spec_avg
                    var.err_r_spec_below = var.r_spec_avg - r_spec_min
                    i_mode += 2
        if ("diff" in kept_modes):
            if not(averaging):
                var.r_diff_reps = saved_var[i_mode]
                i_mode += 1
            if (averaging):
                var.r_diff_avg = saved_var[i_mode]
                i_mode += 1
                if (err):
                    r_diff_min = saved_var[i_mode]
                    r_diff_max = saved_var[i_mode+1]
                    var.err_r_diff_above = r_diff_max - var.r_diff_avg
                    var.err_r_diff_below = var.r_diff_avg - r_diff_min
                    i_mode += 2
        if ("scat" in kept_modes):
            if not(averaging):
                var.r_scat_reps = saved_var[i_mode]
                i_mode += 1
            if (averaging):
                var.r_scat_avg = saved_var[i_mode]
                i_mode += 1
                if (err):
                    r_scat_min = saved_var[i_mode]
                    r_scat_max = saved_var[i_mode+1]
                    var.err_r_scat_above = r_scat_max - var.r_scat_avg
                    var.err_r_scat_below = var.r_scat_avg - r_scat_min
                    i_mode += 2

    if ("tran" in kept_modes):
        if ("spec" in kept_modes):
            if not(averaging):
                var.t_spec_reps = saved_var[i_mode]
                i_mode += 1
            if (averaging):
                var.t_spec_avg = saved_var[i_mode]
                i_mode += 1
                if (err):
                    t_spec_min = saved_var[i_mode]
                    t_spec_max = saved_var[i_mode+1]
                    var.err_t_spec_above = t_spec_max - var.t_spec_avg
                    var.err_t_spec_below = var.t_spec_avg - t_spec_min
                    i_mode += 2

        if ("diff" in kept_modes):
            if not(averaging):
                var.t_diff_reps = saved_var[i_mode]
                i_mode += 1
            if (averaging):
                var.t_diff_avg = saved_var[i_mode]
                i_mode += 1
                if (err):
                    t_diff_min = saved_var[i_mode]
                    t_diff_max = saved_var[i_mode+1]
                    var.err_t_diff_above = t_diff_max - var.t_diff_avg
                    var.err_t_diff_below = var.t_diff_avg - t_diff_min
                    i_mode += 2

        if ("scat" in kept_modes):
            if not(averaging):
                var.t_scat_reps = saved_var[i_mode]
                i_mode += 1
            if (averaging):
                var.t_scat_avg = saved_var[i_mode]
                i_mode += 1
                if (err):
                    t_scat_min = saved_var[i_mode]
                    t_scat_max = saved_var[i_mode+1]
                    var.err_t_scat_above = t_scat_max - var.t_scat_avg
                    var.err_t_scat_below = var.t_scat_avg - t_scat_min
                    i_mode += 2

    if ("abs" in kept_modes):
        if not(averaging):
            var.abs_reps = saved_var[i_mode]
            i_mode += 1
        if (averaging):
            var.abs_avg = saved_var[i_mode]
            i_mode += 1
            if (err):
                abs_min = saved_var[i_mode]
                abs_max = saved_var[i_mode+1]
                var.err_abs_above = abs_max - var.abs_avg
                var.err_abs_below = var.abs_avg - abs_min
                i_mode += 2

    return var


def plot_Lambda(saved_var, kept_modes, variables, params,
                      averaging=True, save=True, path="", file="", subplots=False):
    """
        Plotting 1D plots of Efficiencies (of type kept_modes) against Lambda
    """

    l_structure, l_angle, l_lambdas, l_rf = variables

    err_ev = 8
    # This could be given as a parameter, eventually, but
    # I leave it here for now so that we only have to change it once
    v = load_var(saved_var, kept_modes, averaging)

    for istru in range(len(l_structure)):
        for i_rf in range(len(l_rf)):
            for i_angle in range(len(l_angle)):
                # Make one plot for each of these possibilities
                # (and also for each repetition, if no averaging)

                per = np.round(l_structure[istru].period, 2)
                prof = np.round(l_structure[istru].prof, 2)
                width = np.round(l_structure[istru].interf, 2)
                modes = params.nb_modes
                theta = np.round(l_angle[i_angle].theta, 2)
                phi = np.round(l_angle[i_angle].phi, 2)
                psi = np.round(l_angle[i_angle].psi, 2)
                rf = np.round(l_rf[i_rf], 2)
                # Preparing the file name variables

                if (averaging):
                    nb_reps = params.nb_reps
                    filename = (path + "/" + file + "_per_" + str(per)
                                + "_prof_" + str(prof) + "_width_" + str(width)
                                + "_RF_" + str(rf) + "_theta_" + str(theta)
                                + "_phi_" + str(phi) + "_psi_" + str(psi)
                                + "_nb_reps_" + str(nb_reps) + "_modes_" + str(modes))

                    plt.figure(figsize=(10,10))
                    if ("refl" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.errorbar(l_lambdas, v.r_spec_avg[istru, i_rf, i_angle],
                             yerr=np.array([v.err_r_spec_below[istru, i_rf, i_angle], v.err_r_spec_above[istru, i_rf, i_angle]]),
                             fmt='b', label="Spec. Reflection", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("diff" in kept_modes):
                            plt.errorbar(l_lambdas, v.r_diff_avg[istru, i_rf, i_angle],
                            yerr=np.array([v.err_r_diff_below[istru, i_rf, i_angle], v.err_r_diff_above[istru, i_rf, i_angle]]),
                            fmt='g', label="Upper Diffr. Orders", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("scat" in kept_modes):
                            plt.errorbar(l_lambdas, v.r_scat_avg[istru, i_rf, i_angle],
                            yerr=np.array([v.err_r_scat_below[istru, i_rf, i_angle], v.err_r_scat_above[istru, i_rf, i_angle]]),
                            fmt='r', label="Upper Scattering", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                    if ("tran" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.errorbar(l_lambdas, v.t_spec_avg[istru, i_rf, i_angle],
                            yerr=np.array([v.err_t_spec_below[istru, i_rf, i_angle], v.err_t_spec_above[istru, i_rf, i_angle]]),
                            fmt='b--', label="Spec. Transmission", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("diff" in kept_modes):
                            plt.errorbar(l_lambdas, v.t_diff_avg[istru, i_rf, i_angle],
                            yerr=np.array([v.err_t_diff_below[istru, i_rf, i_angle], v.err_t_diff_above[istru, i_rf, i_angle]]),
                            fmt='g--', label="Lower Diffr. Orders", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("scat" in kept_modes):
                            plt.errorbar(l_lambdas, v.t_scat_avg[istru, i_rf, i_angle],
                            yerr=np.array([v.err_t_scat_below[istru, i_rf, i_angle], v.err_t_scat_above[istru, i_rf, i_angle]]),
                            fmt='r--', label="Lower Scattering", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                    if ("abs" in kept_modes):
                        plt.errorbar(l_lambdas, v.abs_avg[istru, i_rf, i_angle],
                        yerr=np.array([v.err_abs_below[istru, i_rf, i_angle], v.err_abs_above[istru, i_rf, i_angle]]),
                        fmt='k', label="Absorption", errorevery=err_ev, elinewidth=1.0, capsize=2.0)

                    plt.ylim([0,1.1])
                    plt.ylabel("Efficiencies")
                    plt.legend()
                    if (params.unit_lambda_plot=='Hz'):
                        ax1 = plt.gca()
                        ax2 = ax1.twiny()
                        um_lambdas = 299792458/l_lambdas * 1e-6
                        min_int_lambdas = int(um_lambdas[0])
                        max_int_lambdas = int(um_lambdas[-1])
                        int_lambdas = np.arange(min_int_lambdas, max_int_lambdas+1)
                        ax2.set_xticks(1/int_lambdas)
                        ax2.set_xticklabels(str(int_lambdas))
                        ax1.set_xlabel("Frequency (Hz)")
                        ax2.set_xlabel("Wavelength ($\mu$m)")   
                    else:
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                    plt.tight_layout()
                    if (save):
                        plt.savefig(filename + ".svg")
                    print(filename)
                    plt.show()
                    plt.clf()

                else:
                    # Not averaging AND more than one repetition, so one plot
                    # per rep

                    for irep in range(params.nb_reps):

                        filename = (path + "/" + file + "_per_" + str(per)
                                    + "_prof_" + str(prof) + "_width_" + str(width)
                                    + "_RF_" + str(rf) + "_theta_" + str(theta)
                                    + "_phi_" + str(phi) + "_psi_" + str(psi)
                                    + "_irep_" + str(irep) + "_modes_" + str(modes))

                        plt.figure(figsize=(10,10))
                        if ("refl" in kept_modes):
                            if ("spec" in kept_modes):
                                plt.plot(l_lambdas, v.r_spec_reps[istru, i_rf, i_angle, :, irep], 'b', label="Spec. Reflection")
                            if ("diff" in kept_modes):
                                plt.plot(l_lambdas, v.r_diff_reps[istru, i_rf, i_angle, :, irep], 'g', label="Upper Diffr. Orders")
                            if ("scat" in kept_modes):
                                plt.plot(l_lambdas, v.r_scat_reps[istru, i_rf, i_angle, :, irep], 'r', label="Upper Scattering")
                        if ("tran" in kept_modes):
                            if ("spec" in kept_modes):
                                plt.plot(l_lambdas, v.t_spec_reps[istru, i_rf, i_angle, :, irep], 'b--', label="Spec. Transmission")
                            if ("diff" in kept_modes):
                                plt.plot(l_lambdas, v.t_diff_reps[istru, i_rf, i_angle, :, irep], 'g--', label="Lower Diffr. Orders")
                            if ("scat" in kept_modes):
                                plt.plot(l_lambdas, v.t_scat_reps[istru, i_rf, i_angle, :, irep], 'r--', label="Lower Scattering")
                        if ("abs" in kept_modes):
                            plt.plot(l_lambdas, v.abs_reps[istru, i_rf, i_angle, :, irep], 'k', label="Absorption")

                        plt.ylim([0,1.1])
                        plt.ylabel("Efficiencies")
                        if (params.unit_lambda_plot=='Hz'):
                            ax1 = plt.gca()
                            ax2 = ax1.twiny()
                            ax2.set_xlabel("Wavelength ($\mu$m)")
                            ax1.set_xlabel("Frequency (Hz)")
                            um_lambdas = 299792458/l_lambdas * 1e6
                            min_int_lambdas = int(um_lambdas[0])
                            max_int_lambdas = int(um_lambdas[-1])
                            int_lambdas = np.arange(min_int_lambdas, max_int_lambdas+1)
                            str_lambdas = [str(i) for i in int_lambdas]
                            ax2.set_xlim(ax1.get_xlim())
                            ax2.set_xticks(299792458/int_lambdas*1e6)
                            ax2.set_xticklabels(str_lambdas)
                        else:
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.legend()
                        plt.tight_layout()
                        if (save):
                            plt.savefig(filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()

def plot_Theta(saved_var, kept_modes, variables, params,
                      averaging=True, save=True, path="", file="", subplots=False):
    """
        Plotting 1D plots of Efficiencies (of type kept_modes) against Lambda
    """
    print("Warning: for the moment, plotting the evolution of the optical efficiences",
          "against angles considers that only Theta varies.")

    l_structure, l_angle, l_lambdas, l_rf = variables

    l_theta = np.array([l_angle[i].theta for i in range(len(l_angle))])

    err_ev = 2
    # This could be given as a parameter, eventually, but
    # I leave it here for now so that we only have to change it once
    v = load_var(saved_var, kept_modes, averaging)

    for istru in range(len(l_structure)):
        for i_rf in range(len(l_rf)):
            for i_lambda in range(len(l_lambdas)):
                # Make one plot for each of these possibilities
                # (and also for each repetition, if no averaging)

                per = np.round(l_structure[istru].period, 2)
                prof = np.round(l_structure[istru].prof, 2)
                width = np.round(l_structure[istru].interf, 2)
                modes = params.nb_modes
                lambd = np.round(l_lambdas[i_lambda], 2)
                rf = np.round(l_rf[i_rf], 2)
                # Preparing the file name variables

                if (averaging):
                    nb_reps = params.nb_reps
                    filename = (path + "/" + file + "_per_" + str(per)
                                + "_prof_" + str(prof) + "_width_" + str(width)
                                + "_RF_" + str(rf) + "_Lambda_" + str(lambd)
                                + "_nb_reps_" + str(nb_reps) + "_modes_" + str(modes))

                    plt.figure(figsize=(10,10))
                    if ("refl" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.errorbar(l_theta, v.r_spec_avg[istru, i_rf, :, i_lambda],
                             yerr=np.array([v.err_r_spec_below[istru, i_rf, :, i_lambda], v.err_r_spec_above[istru, i_rf, :, i_lambda]]),
                             fmt='b', label="Spec. Reflection", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("diff" in kept_modes):
                            plt.errorbar(l_theta, v.r_diff_avg[istru, i_rf, :, i_lambda],
                            yerr=np.array([v.err_r_diff_below[istru, i_rf, :, i_lambda], v.err_r_diff_above[istru, i_rf, :, i_lambda]]),
                            fmt='g', label="Upper Diffr. Orders", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("scat" in kept_modes):
                            plt.errorbar(l_theta, v.r_scat_avg[istru, i_rf, :, i_lambda],
                            yerr=np.array([v.err_r_scat_below[istru, i_rf, :, i_lambda], v.err_r_scat_above[istru, i_rf, :, i_lambda]]),
                            fmt='r', label="Upper Scattering", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                    if ("tran" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.errorbar(l_theta, v.t_spec_avg[istru, i_rf, :, i_lambda],
                            yerr=np.array([v.err_t_spec_below[istru, i_rf, :, i_lambda], v.err_t_spec_above[istru, i_rf, :, i_lambda]]),
                            fmt='b--', label="Spec. Transmission", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("diff" in kept_modes):
                            plt.errorbar(l_theta, v.t_diff_avg[istru, i_rf, :, i_lambda],
                            yerr=np.array([v.err_t_diff_below[istru, i_rf, :, i_lambda], v.err_t_diff_above[istru, i_rf, :, i_lambda]]),
                            fmt='g--', label="Lower Diffr. Orders", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("scat" in kept_modes):
                            plt.errorbar(l_theta, v.t_scat_avg[istru, i_rf, :, i_lambda],
                            yerr=np.array([v.err_t_scat_below[istru, i_rf, :, i_lambda], v.err_t_scat_above[istru, i_rf, :, i_lambda]]),
                            fmt='r--', label="Lower Scattering", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                    if ("abs" in kept_modes):
                        plt.errorbar(l_theta, v.abs_avg[istru, i_rf, :, i_lambda],
                        yerr=np.array([v.err_abs_below[istru, i_rf, :, i_lambda], v.err_abs_above[istru, i_rf, :, i_lambda]]),
                        fmt='k', label="Absorption", errorevery=err_ev, elinewidth=1.0, capsize=2.0)

                    plt.ylim([0,1.1])
                    plt.xlabel("Incident Angle ({})".format(params.unit_angle_plot))
                    plt.ylabel("Efficiencies")
                    plt.legend()
                    plt.tight_layout()
                    if (save):
                        plt.savefig(filename + ".svg")
                    print(filename)
                    plt.show()
                    plt.clf()

                else:
                    # Not averaging AND more than one repetition, so one plot
                    # per rep

                    for irep in range(params.nb_reps):

                        filename = (path + "/" + file + "_per_" + str(per)
                                    + "_prof_" + str(prof) + "_width_" + str(width)
                                    + "_RF_" + str(rf) + "_Lambda_" + str(lambd)
                                    + "_irep_" + str(irep) + "_modes_" + str(modes))

                        plt.figure(figsize=(10,10))
                        if ("refl" in kept_modes):
                            if ("spec" in kept_modes):
                                print(np.shape(v.r_spec_reps))
                                plt.plot(l_theta, v.r_spec_reps[istru, i_rf, :, i_lambda, irep], 'b', label="Spec. Reflection")
                            if ("diff" in kept_modes):
                                plt.plot(l_theta, v.r_diff_reps[istru, i_rf, :, i_lambda, irep], 'g', label="Upper Diffr. Orders")
                            if ("scat" in kept_modes):
                                plt.plot(l_theta, v.r_scat_reps[istru, i_rf, :, i_lambda, irep], 'r', label="Upper Scattering")
                        if ("tran" in kept_modes):
                            if ("spec" in kept_modes):
                                plt.plot(l_theta, v.t_spec_reps[istru, i_rf, :, i_lambda, irep], 'b--', label="Spec. Transmission")
                            if ("diff" in kept_modes):
                                plt.plot(l_theta, v.t_diff_reps[istru, i_rf, :, i_lambda, irep], 'g--', label="Lower Diffr. Orders")
                            if ("scat" in kept_modes):
                                plt.plot(l_theta, v.t_scat_reps[istru, i_rf, :, i_lambda, irep], 'r--', label="Lower Scattering")
                        if ("abs" in kept_modes):
                            plt.plot(l_theta, v.abs_reps[istru, i_rf, :, i_lambda, irep], 'k', label="Absorption")

                        plt.ylim([0,1.1])
                        plt.xlabel("Incident Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Efficiencies")
                        plt.legend()
                        plt.tight_layout()
                        if (save):
                            plt.savefig(filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()

def plot_RF(saved_var, kept_modes, variables, params,
                      averaging=True, save=True, path="", file="", subplots=False):
    """
        Plotting 1D plots of Efficiencies (of type kept_modes) against Lambda
    """

    l_structure, l_angle, l_lambdas, l_rf = variables

    err_ev = 2
    # This could be given as a parameter, eventually, but
    # I leave it here for now so that we only have to change it once
    v = load_var(saved_var, kept_modes, averaging)

    for istru in range(len(l_structure)):
        for i_angle in range(len(l_angle)):
            for i_lambda in range(len(l_lambdas)):
                # Make one plot for each of these possibilities
                # (and also for each repetition, if no averaging)

                per = np.round(l_structure[istru].period, 2)
                prof = np.round(l_structure[istru].prof, 2)
                width = np.round(l_structure[istru].interf, 2)
                modes = params.nb_modes
                theta = np.round(l_angle[i_angle].theta, 2)
                phi = np.round(l_angle[i_angle].phi, 2)
                psi = np.round(l_angle[i_angle].psi, 2)
                lambd = np.round(l_lambdas[i_lambda], 2)
                # Preparing the file name variables

                if (averaging):
                    nb_reps = params.nb_reps
                    filename = (path + "/" + file + "_per_" + str(per)
                                + "_prof_" + str(prof) + "_width_" + str(width)
                                + "_Lambda_" + str(lambd) + "_theta_" + str(theta)
                                + "_phi_" + str(phi) + "_psi_" + str(psi)
                                + "_nb_reps_" + str(nb_reps) + "_modes_" + str(modes))

                    plt.figure(figsize=(10,10))
                    if ("refl" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.errorbar(l_rf, v.r_spec_avg[istru, :, i_angle, i_lambda],
                             yerr=np.array([v.err_r_spec_below[istru, :, i_angle, i_lambda], v.err_r_spec_above[istru, :, i_angle, i_lambda]]),
                             fmt='b', label="Spec. Reflection", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("diff" in kept_modes):
                            plt.errorbar(l_rf, v.r_diff_avg[istru, :, i_angle, i_lambda],
                            yerr=np.array([v.err_r_diff_below[istru, :, i_angle, i_lambda], v.err_r_diff_above[istru, :, i_angle, i_lambda]]),
                            fmt='g', label="Upper Diffr. Orders", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("scat" in kept_modes):
                            plt.errorbar(l_rf, v.r_scat_avg[istru, :, i_angle, i_lambda],
                            yerr=np.array([v.err_r_scat_below[istru, :, i_angle, i_lambda], v.err_r_scat_above[istru, :, i_angle, i_lambda]]),
                            fmt='r', label="Upper Scattering", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                    if ("tran" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.errorbar(l_rf, v.t_spec_avg[istru, :, i_angle, i_lambda],
                            yerr=np.array([v.err_t_spec_below[istru, :, i_angle, i_lambda], v.err_t_spec_above[istru, :, i_angle, i_lambda]]),
                            fmt='b--', label="Spec. Transmission", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("diff" in kept_modes):
                            plt.errorbar(l_rf, v.t_diff_avg[istru, :, i_angle, i_lambda],
                            yerr=np.array([v.err_t_diff_below[istru, :, i_angle, i_lambda], v.err_t_diff_above[istru, :, i_angle, i_lambda]]),
                            fmt='g--', label="Lower Diffr. Orders", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                        if ("scat" in kept_modes):
                            plt.errorbar(l_rf, v.t_scat_avg[istru, :, i_angle, i_lambda],
                            yerr=np.array([v.err_t_scat_below[istru, :, i_angle, i_lambda], v.err_t_scat_above[istru, :, i_angle, i_lambda]]),
                            fmt='r--', label="Lower Scattering", errorevery=err_ev, elinewidth=1.0, capsize=2.0)
                    if ("abs" in kept_modes):
                        plt.errorbar(l_rf, v.abs_avg[istru, :, i_angle, i_lambda],
                        yerr=np.array([v.err_abs_below[istru, :, i_angle, i_lambda], v.err_abs_above[istru, :, i_angle, i_lambda]]),
                        fmt='k', label="Absorption", errorevery=err_ev, elinewidth=1.0, capsize=2.0)

                    plt.ylim([0,1.1])
                    plt.xlabel("Random Factor ({})".format(params.unit_rf_plot))
                    plt.ylabel("Efficiencies")
                    plt.legend()
                    plt.tight_layout()
                    if (save):
                        plt.savefig(filename + ".svg")
                    print(filename)
                    plt.show()
                    plt.clf()

                else:
                    # Not averaging AND more than one repetition, so one plot
                    # per rep

                    for irep in range(params.nb_reps):

                        filename = (path + "/" + file + "_per_" + str(per)
                                    + "_prof_" + str(prof) + "_width_" + str(width)
                                    + "_Lambda_" + str(lambd) + "_theta_" + str(theta)
                                    + "_phi_" + str(phi) + "_psi_" + str(psi)
                                    + "_irep_" + str(irep) + "_modes_" + str(modes))

                        plt.figure(figsize=(10,10))
                        if ("refl" in kept_modes):
                            if ("spec" in kept_modes):
                                plt.plot(l_rf, v.r_spec_reps[istru, :, i_angle, i_lambda, irep], 'b', label="Spec. Reflection")
                            if ("diff" in kept_modes):
                                plt.plot(l_rf, v.r_diff_reps[istru, :, i_angle, i_lambda, irep], 'g', label="Upper Diffr. Orders")
                            if ("scat" in kept_modes):
                                plt.plot(l_rf, v.r_scat_reps[istru, :, i_angle, i_lambda, irep], 'r', label="Upper Scattering")
                        if ("tran" in kept_modes):
                            if ("spec" in kept_modes):
                                plt.plot(l_rf, v.t_spec_reps[istru, :, i_angle, i_lambda, irep], 'b--', label="Spec. Transmission")
                            if ("diff" in kept_modes):
                                plt.plot(l_rf, v.t_diff_reps[istru, :, i_angle, i_lambda, irep], 'g--', label="Lower Diffr. Orders")
                            if ("scat" in kept_modes):
                                plt.plot(l_rf, v.t_scat_reps[istru, :, i_angle, i_lambda, irep], 'r--', label="Lower Scattering")
                        if ("abs" in kept_modes):
                            plt.plot(l_rf, v.abs_reps[istru, :, i_angle, i_lambda, irep], 'k', label="Absorption")



def plot_LambdaTheta(saved_var, kept_modes, variables, params,
              averaging, save, path, file, subplots=False, contours=0):
    """
        2D plot (plt.pcolor) Wavelength/Incidence Angle
        Here, only one outgoing field is plotted on each graph
    """

    l_structure, l_angle, l_lambdas, l_rf = variables

    v = load_var(saved_var, kept_modes, averaging, err=False)

    for istru in range(len(l_structure)):
        for irf in range(len(l_rf)):

            per = np.round(l_structure[istru].period, 2)
            prof = np.round(l_structure[istru].prof, 2)
            width = np.round(l_structure[istru].interf, 2)
            modes = params.nb_modes
            rf = np.round(l_rf[irf], 2)
            # Preparing the file name variables

            l_thetas = np.array([l_angle[i].theta for i in range(len(l_angle))])
            X, Y = np.meshgrid(l_lambdas, l_thetas)

            if (averaging):
                nb_reps = params.nb_reps
                filename = (file + "_per_" + str(per)
                            + "_prof_" + str(prof) + "_width_" + str(width)
                            + "_rf_" + str(rf) + "_nbreps_" + str(nb_reps)
                            + "_modes_" + str(modes))

                if ("refl" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_spec_avg[istru, irf], cmap='viridis')
                        if (contours):
                            plt.contour(X, Y, v.r_spec_avg[istru, irf], contours, colors="k")
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_diff_avg[istru, irf], cmap='viridis')
                        if (contours):
                            plt.contour(X, Y, v.r_diff_avg[istru, irf], contours, colors="k")
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_scat_avg[istru, irf], cmap='viridis')
                        if (contours):
                            plt.contour(X, Y, v.r_scat_avg[istru, irf], contours, colors="k")
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                if ("tran" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_spec_avg[istru, irf], cmap='viridis')
                        if (contours):
                            plt.contour(X, Y, v.t_spec_avg[istru, irf], contours, colors="k")
                        
                        if (params.unit_lambda_plot=='Hz'):
                            ax1 = plt.gca()
                            ax2 = ax1.twiny()
                            ax2.set_xlabel("Wavelength ($\mu$m)")
                            ax1.set_xlabel("Frequency (Hz)")
                            um_lambdas = 299792458/l_lambdas * 1e6
                            min_int_lambdas = int(um_lambdas[0])
                            max_int_lambdas = int(um_lambdas[-1])
                            int_lambdas = np.arange(min_int_lambdas, max_int_lambdas+1)
                            str_lambdas = [str(i) for i in int_lambdas]
                            ax2.set_xlim(ax1.get_xlim())
                            ax2.set_xticks(299792458/int_lambdas*1e6)
                            ax2.set_xticklabels(str_lambdas)
                        else:
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_diff_avg[istru, irf], cmap='viridis')
                        if (contours):
                            plt.contour(X, Y, v.t_diff_avg[istru, irf], contours, colors="k")
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_scat_avg[istru, irf], cmap='viridis')
                        if (contours):
                            plt.contour(X, Y, v.t_scat_avg[istru, irf], contours, colors="k")
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                if ("abs" in kept_modes):
                    plt.figure(figsize=(10,10))
                    CS = plt.pcolormesh(X, Y, v.abs_avg[istru, irf], cmap='viridis')
                    if (contours):
                        plt.contour(X, Y, v.abs_avg[istru, irf], contours, colors="k")
                    plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                    plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                    plt.colorbar(CS)
                    if (save):
                        plt.savefig(path + "/Abs_" + filename + ".svg")
                    print(filename)
                    plt.show()
                    plt.clf()

            if not(averaging):
                for irep in range(params.nb_reps):
                    filename = (file + "_per_" + str(per)
                                + "_prof_" + str(prof) + "_width_" + str(width)
                                + "_rf_" + str(rf) + "_irep_" + str(irep)
                                + "_modes_" + str(modes))

                    if ("refl" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_spec_reps[istru, irf, :, :, irep], cmap='viridis')
                            if (contours):
                                plt.contour(X, Y, v.r_spec_reps[istru, irf, :, :, irep], contours, colors="k")
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_diff_reps[istru, irf, :, :, irep], cmap='viridis')
                            if (contours):
                                plt.contour(X, Y, v.r_diff_reps[istru, irf, :, :, irep], contours, colors="k")
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_scat_reps[istru, irf, :, :, irep], cmap='viridis')
                            if (contours):
                                plt.contour(X, Y, v.r_scat_reps[istru, irf, :, :, irep], contours, colors="k")
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                    if ("tran" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_spec_reps[istru, irf, :, :, irep], cmap='viridis')
                            if (contours):
                                plt.contour(X, Y, v.t_spec_reps[istru, irf, :, :, irep], contours, colors="k")
                            
                            if (params.unit_lambda_plot=='Hz'):
                                ax1 = plt.gca()
                                ax2 = ax1.twiny()
                                ax2.set_xlabel("Wavelength ($\mu$m)")
                                ax1.set_xlabel("Frequency (Hz)")
                                um_lambdas = 299792458/l_lambdas * 1e6
                                min_int_lambdas = int(um_lambdas[0])
                                max_int_lambdas = int(um_lambdas[-1])
                                int_lambdas = np.arange(min_int_lambdas, max_int_lambdas+1)
                                str_lambdas = [str(i) for i in int_lambdas]
                                ax2.set_xlim(ax1.get_xlim())
                                ax2.set_xticks(299792458/int_lambdas*1e6)
                                ax2.set_xticklabels(str_lambdas)
                            else:
                                plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_diff_reps[istru, irf, :, :, irep], cmap='viridis')
                            if (contours):
                                plt.contour(X, Y, v.t_diff_reps[istru, irf, :, :, irep], contours, colors="k")
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_scat_reps[istru, irf, :, :, irep], cmap='viridis')
                            if (contours):
                                plt.contour(X, Y, v.t_scat_reps[istru, irf, :, :, irep], contours, colors="k")
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                    if ("abs" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.abs_reps[istru, irf, :, :, irep], cmap='viridis')
                        if (contours):
                            plt.contour(X, Y, v.abs_reps[istru, irf, :, :, irep], contours, colors="k")
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Incidence Angle({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Abs_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()


def plot_SigmaKx(saved_var, kept_modes, variables, params,
              averaging, save, path, file, subplots=False):
    """
        2D plot (plt.pcolor) Wavelength/Incidence Angle
        Here, only one outgoing field is plotted on each graph
    """

    l_structure, l_angle, l_lambdas, l_rf = variables

    v = load_var(saved_var, kept_modes, averaging, err=False)

    for istru in range(len(l_structure)):
        for irf in range(len(l_rf)):

            per = np.round(l_structure[istru].period, 2)
            prof = np.round(l_structure[istru].prof, 2)
            width = np.round(l_structure[istru].interf, 2)
            modes = params.nb_modes
            rf = np.round(l_rf[irf], 2)
            # Preparing the file name variables

            l_thetas = np.array([l_angle[i].theta for i in range(len(l_angle))])
            X, Y = np.meshgrid(l_thetas, l_lambdas)
            if (params.unit_angle_plot == "kx" and params.unit_lambda_plot == "cm-1"):
                for i in range(len(l_thetas)):
                    X[:,i] = X[:,i] * l_lambdas

            if (averaging):
                nb_reps = params.nb_reps
                filename = (file + "_per_" + str(per)
                            + "_prof_" + str(prof) + "_width_" + str(width)
                            + "_rf_" + str(rf) + "_nbreps_" + str(nb_reps)
                            + "_modes_" + str(modes))

                if ("refl" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_spec_avg[istru, irf].T, cmap='viridis')
                        plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_diff_avg[istru, irf].T, cmap='viridis')
                        plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_scat_avg[istru, irf].T, cmap='viridis')
                        plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                if ("tran" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_spec_avg[istru, irf].T, cmap='viridis')
                        plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_diff_avg[istru, irf].T, cmap='viridis')
                        plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_scat_avg[istru, irf].T, cmap='viridis')
                        plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                if ("abs" in kept_modes):
                    plt.figure(figsize=(10,10))
                    CS = plt.pcolormesh(X, Y, v.abs_avg[istru, irf].T, cmap='viridis')
                    plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                    plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                    plt.colorbar(CS)
                    if (save):
                        plt.savefig(path + "/Abs_" + filename + ".svg")
                    print(filename)
                    plt.show()
                    plt.clf()

            if not(averaging):
                for irep in range(params.nb_reps):
                    filename = (file + "_per_" + str(per)
                                + "_prof_" + str(prof) + "_width_" + str(width)
                                + "_rf_" + str(rf) + "_irep_" + str(irep)
                                + "_modes_" + str(modes))

                    if ("refl" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_spec_reps[istru, irf, :, :, irep].T, cmap='viridis')
                            plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_diff_reps[istru, irf, :, :, irep].T, cmap='viridis')
                            plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_scat_reps[istru, irf, :, :, irep].T, cmap='viridis')
                            plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                    if ("tran" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_spec_reps[istru, irf, :, :, irep].T, cmap='viridis')
                            plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_diff_reps[istru, irf, :, :, irep].T, cmap='viridis')
                            plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_scat_reps[istru, irf, :, :, irep].T, cmap='viridis')
                            plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                    if ("abs" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.abs_reps[istru, irf, :, :, irep].T, cmap='viridis')
                        plt.ylabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Abs_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                        
                        
def plot_LambdaRF(saved_var, kept_modes, variables, params,
              averaging, save, path, file, subplots=False):
    """
        2D plot (plt.pcolor) Wavelength/Random Factor
        Here, only one outgoing field is plotted on each graph
    """

    l_structure, l_angle, l_lambdas, l_rf = variables

    v = load_var(saved_var, kept_modes, averaging, err=False)

    for istru in range(len(l_structure)):
        for iangle in range(len(l_angle)):

            per = np.round(l_structure[istru].period, 2)
            prof = np.round(l_structure[istru].prof, 2)
            width = np.round(l_structure[istru].interf, 2)
            modes = params.nb_modes
            theta = np.round(l_angle[iangle].theta, 2)
            phi = np.round(l_angle[iangle].phi, 2)
            psi = np.round(l_angle[iangle].psi, 2)
            # Preparing the file name variables

            X, Y = np.meshgrid(l_lambdas, l_rf)

            if (averaging):
                nb_reps = params.nb_reps
                filename = (file + "_per_" + str(per)
                            + "_prof_" + str(prof) + "_width_" + str(width)
                            + "_theta_" + str(theta) + "_phi_" + str(phi)
                            + "_psi_" + str(psi) + "_nbreps_" + str(nb_reps)
                            + "_modes_" + str(modes))

                if ("refl" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_spec_avg[istru, :, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                        print(filename+"_refl_spec")
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_diff_avg[istru, :, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                        print(filename+"_refl_diff")
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_scat_avg[istru, :, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                        print(filename+"_refl_scat")
                        plt.show()
                        plt.clf()
                if ("tran" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_spec_avg[istru, :, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                        print(filename+"_tran_spec")
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_diff_avg[istru, :, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                        print(filename+"_tran_diff")
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_scat_avg[istru, :, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                        print(filename+"_tran_scat")
                        plt.show()
                        plt.clf()
                if ("abs" in kept_modes):
                    plt.figure(figsize=(10,10))
                    CS = plt.pcolormesh(X, Y, v.abs_avg[istru, :, iangle, :], cmap='viridis')
                    plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                    plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                    plt.colorbar(CS)
                    if (save):
                        plt.savefig(path + "/Abs_" + filename + ".svg")
                    print(filename+"_abs")
                    plt.show()
                    plt.clf()

            if not(averaging):
                for irep in range(params.nb_reps):
                    filename = (file + "_per_" + str(per)
                                + "_prof_" + str(prof) + "_width_" + str(width)
                                + "_theta_" + str(theta) + "_phi_" + str(phi)
                                + "_psi_" + str(psi)+ "_irep_" + str(irep)
                                + "_modes_" + str(modes))

                    if ("refl" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_spec_reps[istru, :, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                            print(filename+"_refl_spec")
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_diff_reps[istru, :, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                            print(filename+"_refl_diff")
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_scat_reps[istru, :, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                            print(filename+"_refl_scat")
                            plt.show()
                            plt.clf()
                    if ("tran" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_spec_reps[istru, :, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                            print(filename+"_tran_spec")
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_diff_reps[istru, :, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                            print(filename+"_tran_diff")
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_scat_reps[istru, :, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.legend()
                            plt.tight_layout()
                            if (save):
                                plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                            print(filename+"_tran_scat")
                            plt.show()
                            plt.clf()
                    if ("abs" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.abs_reps[istru, :, iangle, :, irep], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Abs_" + filename + ".svg")
                        print(filename+"_abs")
                        plt.show()
                        plt.clf()

def plot_ThetaRF(saved_var, kept_modes, variables, params,
              averaging, save, path, file, subplots=False):
    """
        2D plot (plt.pcolor) Wavelength/Random Factor
        Here, only one outgoing field is plotted on each graph
    """

    l_structure, l_angle, l_lambdas, l_rf = variables

    v = load_var(saved_var, kept_modes, averaging, err=False)

    for istru in range(len(l_structure)):
        for ilamb in range(len(l_lambdas)):

            per = np.round(l_structure[istru].period, 2)
            prof = np.round(l_structure[istru].prof, 2)
            width = np.round(l_structure[istru].interf, 2)
            modes = params.nb_modes
            lamb = np.round(l_lambdas[ilamb], 2)
            # Preparing the file name variables

            l_thetas = np.array([l_angle[i].theta for i in range(len(l_angle))])
            X, Y = np.meshgrid(l_thetas, l_rf)

            if (averaging):
                nb_reps = params.nb_reps
                filename = (file + "_per_" + str(per)
                            + "_prof_" + str(prof) + "_width_" + str(width)
                            + "_lambda_" + str(lamb) + "_nbreps_" + str(nb_reps)
                            + "_modes_" + str(modes))

                if ("refl" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_spec_avg[istru, :, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                        print(filename+"_refl_spec")
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_diff_avg[istru, :, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                        print(filename+"_refl_diff")
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_scat_avg[istru, :, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                        print(filename+"_refl_scat")
                        plt.show()
                        plt.clf()
                if ("tran" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_spec_avg[istru, :, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                        print(filename+"_tran_spec")
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_diff_avg[istru, :, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                        print(filename+"_tran_diff")
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_scat_avg[istru, :, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                        print(filename+"_tran_scat")
                        plt.show()
                        plt.clf()
                if ("abs" in kept_modes):
                    plt.figure(figsize=(10,10))
                    CS = plt.pcolormesh(X, Y, v.abs_avg[istru, :, :, ilamb], cmap='viridis')
                    plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                    plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                    plt.colorbar(CS)
                    if (save):
                        plt.savefig(path + "/Abs_" + filename + ".svg")
                    print(filename+"_abs")
                    plt.show()
                    plt.clf()

            if not(averaging):
                for irep in range(params.nb_reps):
                    filename = (file + "_per_" + str(per)
                                + "_prof_" + str(prof) + "_width_" + str(width)
                                + "_lambda_" + str(lamb) + "_irep_" + str(irep)
                                + "_modes_" + str(modes))

                    if ("refl" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_spec_reps[istru, :, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                            print(filename+"_refl_spec")
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_diff_reps[istru, :, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                            print(filename+"_refl_diff")
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_scat_reps[istru, :, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                            print(filename+"_refl_scat")
                            plt.show()
                            plt.clf()
                    if ("tran" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_spec_reps[istru, :, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                            print(filename+"_tran_spec")
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_diff_reps[istru, :, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                            print(filename+"_tran_diff")
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_scat_reps[istru, :, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                            plt.legend()
                            plt.tight_layout()
                            if (save):
                                plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                            print(filename+"_tran_scat")
                            plt.show()
                            plt.clf()
                    if ("abs" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.abs_reps[istru, :, :, ilamb, irep], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Random Factor({})".format(params.unit_rf_plot))
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Abs_" + filename + ".svg")
                        print(filename+"_abs")
                        plt.show()
                        plt.clf()



def plot_LambdaStruct(saved_var, kept_modes, variables, params,
              averaging, save, path, file, subplots=False):
    """
        2D plot (plt.pcolor) Wavelength/Random Factor
        Here, only one outgoing field is plotted on each graph
    """

    l_structure, l_angle, l_lambdas, l_rf = variables

    v = load_var(saved_var, kept_modes, averaging, err=False)

    for iangle in range(len(l_angle)):
        for irf in range(len(l_rf)):

            rf = np.round(l_rf[irf], 2)
            modes = params.nb_modes
            theta = np.round(l_angle[iangle].theta, 2)
            phi = np.round(l_angle[iangle].phi, 2)
            psi = np.round(l_angle[iangle].psi, 2)
            # Preparing the file name variables

            X, Y = np.meshgrid(l_lambdas, np.arange(len(l_structure)))

            if (averaging):
                nb_reps = params.nb_reps
                filename = (file + "Lambda_Struct_Plot_" + "rf_" + str(rf)
                            + "_theta_" + str(theta) + "_phi_" + str(phi)
                            + "_psi_" + str(psi) + "_nbreps_" + str(nb_reps)
                            + "_modes_" + str(modes))

                if ("refl" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_spec_avg[:, irf, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_diff_avg[:, irf, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_scat_avg[:, irf, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                if ("tran" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_spec_avg[:, irf, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_diff_avg[:, irf, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_scat_avg[:, irf, iangle, :], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                if ("abs" in kept_modes):
                    plt.figure(figsize=(10,10))
                    CS = plt.pcolormesh(X, Y, v.abs_avg[:, irf, iangle, :], cmap='viridis')
                    plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                    plt.ylabel("Structure nb")
                    plt.colorbar(CS)
                    if (save):
                        plt.savefig(path + "/Abs_" + filename + ".svg")
                    print(filename)
                    plt.show()
                    plt.clf()

            if not(averaging):
                for irep in range(params.nb_reps):
                    filename = (file + "Lambda_Struct_Plot_" + "rf_" + str(rf)
                                + "_theta_" + str(theta) + "_phi_" + str(phi)
                                + "_psi_" + str(psi)+ "_irep_" + str(irep)
                                + "_modes_" + str(modes))

                    if ("refl" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_spec_reps[:, irf, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_diff_reps[:, irf, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_scat_reps[:, irf, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                    if ("tran" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_spec_reps[:, irf, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_diff_reps[:, irf, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_scat_reps[:, irf, iangle, :, irep], cmap='viridis')
                            plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                            plt.ylabel("Structure nb")
                            plt.legend()
                            plt.tight_layout()
                            if (save):
                                plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                    if ("abs" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.abs_reps[:, irf, iangle, :, irep], cmap='viridis')
                        plt.xlabel("Wavelength ({})".format(params.unit_lambda_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Abs_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()


def plot_ThetaStruct(saved_var, kept_modes, variables, params,
              averaging, save, path, file, subplots=False):
    """
        2D plot (plt.pcolor) Wavelength/Random Factor
        Here, only one outgoing field is plotted on each graph
    """

    l_structure, l_angle, l_lambdas, l_rf = variables

    v = load_var(saved_var, kept_modes, averaging, err=False)

    for irf in range(len(l_rf)):
        for ilamb in range(len(l_lambdas)):

            rf = np.round(l_rf[irf], 2)
            modes = params.nb_modes
            lamb = np.round(l_lambdas[ilamb], 2)
            # Preparing the file name variables

            l_thetas = np.array([l_angle[i].theta for i in range(len(l_angle))])
            X, Y = np.meshgrid(l_thetas, np.arange(len(l_structure)))

            if (averaging):
                nb_reps = params.nb_reps
                filename = (file + "Theta_Struct_Plot_" + "rf_" + str(rf)
                            + "_lambda_" + str(lamb) + "_nbreps_" + str(nb_reps)
                            + "_modes_" + str(modes))

                if ("refl" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_spec_avg[:, irf, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_diff_avg[:, irf, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.r_scat_avg[:, irf, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                if ("tran" in kept_modes):
                    if ("spec" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_spec_avg[:, irf, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("diff" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_diff_avg[:, irf, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                    if ("scat" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.t_scat_avg[:, irf, :, ilamb], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                if ("abs" in kept_modes):
                    plt.figure(figsize=(10,10))
                    CS = plt.pcolormesh(X, Y, v.abs_avg[:, irf, :, ilamb], cmap='viridis')
                    plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                    plt.ylabel("Structure nb")
                    plt.colorbar(CS)
                    if (save):
                        plt.savefig(path + "/Abs_" + filename + ".svg")
                    print(filename)
                    plt.show()
                    plt.clf()

            if not(averaging):
                for irep in range(params.nb_reps):
                    filename = (file + "Lambda_Struct_Plot_" + "rf_" + str(rf)
                                + "_lambda_" + str(lamb) + "_irep_" + str(irep)
                                + "_modes_" + str(modes))

                    if ("refl" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_spec_reps[:, irf, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_diff_reps[:, irf, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.r_scat_reps[:, irf, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Scat_Refl_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                    if ("tran" in kept_modes):
                        if ("spec" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_spec_reps[:, irf, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Spec_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("diff" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_diff_reps[:, irf, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Structure nb")
                            plt.colorbar(CS)
                            if (save):
                                plt.savefig(path + "/Diff_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                        if ("scat" in kept_modes):
                            plt.figure(figsize=(10,10))
                            CS = plt.pcolormesh(X, Y, v.t_scat_reps[:, irf, :, ilamb, irep], cmap='viridis')
                            plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                            plt.ylabel("Structure nb")
                            plt.legend()
                            plt.tight_layout()
                            if (save):
                                plt.savefig(path + "/Scat_Trans_" + filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()
                    if ("abs" in kept_modes):
                        plt.figure(figsize=(10,10))
                        CS = plt.pcolormesh(X, Y, v.abs_reps[:, irf, :, ilamb, irep], cmap='viridis')
                        plt.xlabel("Incidence Angle ({})".format(params.unit_angle_plot))
                        plt.ylabel("Structure nb")
                        plt.colorbar(CS)
                        if (save):
                            plt.savefig(path + "/Abs_" + filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()



def load_scat(saved_var, kept_modes, averaging, err=False):
    """
        Load the necessary variables for plotting
        - err tells the function whether or not to load the error bars
    """

    var = BMM.Bunch()
    i_mode = 0
    print(len(saved_var))
    # This will serve as an index in the saved_var list
    # that will increase depending on kept_modes
    if ("refl" in kept_modes):
        if not(averaging):
            var.r_out_reps = saved_var[i_mode]
            i_mode += 1
        if (averaging):
            var.r_out_avg = saved_var[i_mode]
            i_mode += 1
            if (err):
                r_out_min = saved_var[i_mode]
                r_out_max = saved_var[i_mode+1]
                var.err_r_out_above = r_out_max - var.r_out_avg
                var.err_r_out_below = var.r_out_avg - r_out_min
                i_mode += 2
        var.r_out_angl = saved_var[i_mode]
        i_mode += 1
       
    if ("tran" in kept_modes):
        if not(averaging):
            var.t_out_reps = saved_var[i_mode]
            i_mode += 1
        if (averaging):
            var.t_out_avg = saved_var[i_mode]
            i_mode += 1
            if (err):
                t_out_min = saved_var[i_mode]
                t_out_max = saved_var[i_mode+1]
                var.err_t_out_above = t_out_max - var.t_out_avg
                var.err_t_out_below = var.t_out_avg - t_out_min
                i_mode += 2
        var.t_out_angl = saved_var[i_mode]
        i_mode += 1

    return var


def plot_Scat(saved_var, kept_modes, variables, params, variance,
                      averaging=True, save=True, path="", file="", subplots=False):
    """
        Plotting 1D plots of Efficiencies (of type kept_modes) against Lambda
    """
    l_structure, l_angle, l_lambdas, l_rf = variables

    v = load_scat(saved_var, kept_modes, averaging, err=False)

    for istru in range(len(l_structure)):
        for i_rf in range(len(l_rf)):
            for i_angle in range(len(l_angle)):
                for i_lam in range(len(l_lambdas)):
                    # Make one plot for each of these possibilities
                    # (and also for each repetition, if no averaging)
    
                    per = np.round(l_structure[istru].period, 2)
                    prof = np.round(l_structure[istru].prof, 2)
                    width = np.round(l_structure[istru].interf, 2)
                    modes = params.nb_modes
                    theta = np.round(l_angle[i_angle].theta, 2)
                    phi = np.round(l_angle[i_angle].phi, 2)
                    psi = np.round(l_angle[i_angle].psi, 2)
                    rf = np.round(l_rf[i_rf], 2)
                    lamb = np.round(l_lambdas[i_lam], 2)
                    var = np.round(variance, 2)
                    # Preparing the file name variables
                    if (averaging):
                        nb_reps = params.nb_reps
                        file_root = (file + "_per_" + str(per)
                                    + "_prof_" + str(prof) + "_width_" + str(width)
                                    + "_RF_" + str(rf) + "_VAR_" + str(var) + "_theta_" + str(theta)
                                    + "_phi_" + str(phi) + "_psi_" + str(psi) + "_lamb_" + str(lamb)
                                    + "_nb_reps_" + str(nb_reps) + "_modes_" + str(modes))
                        filename = (path + "/" + file + "_per_" + str(per)
                                    + "_prof_" + str(prof) + "_width_" + str(width)
                                    + "_RF_" + str(rf) + "_VAR_" + str(var) + "_theta_" + str(theta)
                                    + "_phi_" + str(phi) + "_psi_" + str(psi) + "_lamb_" + str(lamb)
                                    + "_nb_reps_" + str(nb_reps) + "_modes_" + str(modes))
    
                        plt.figure(figsize=(10,10))

                        if ("refl" in kept_modes):
                            refl_modes = v.r_out_angl[istru, i_rf, i_angle, i_lam]
                            k0 = 2*np.pi / l_lambdas[i_lam]
                            kx0 = k0 * np.sin(l_angle[i_angle].theta*np.pi/180)
                            kx_per = 2*np.pi/(l_structure[istru].period*params.super_period)
                            refl_k = kx0 + refl_modes*kx_per
                            r_angl_out = np.arcsin(refl_k / (2*np.pi / l_lambdas[i_lam]))*180/np.pi
#                            print(refl_modes, kx0, kx_per, r_angl_out)
                            plt.plot(r_angl_out, v.r_out_avg[istru, i_rf, i_angle, i_lam], 'b', label="out. Reflection")
                           
                        if ("tran" in kept_modes):
                            tran_modes = v.t_out_angl[istru, i_rf, i_angle, i_lam]
                            k0 = 2*np.pi / lamb
                            kx0 = k0 * np.sin(l_angle[i_angle].theta)
                            kx_per = 2*np.pi/(l_structure[istru].period*params.super_period)
                            tran_k = kx0 + tran_modes*kx_per
                            t_angl_out = np.arcsin(tran_k / 2*np.pi / lamb)*180/np.pi
                            plt.plot(t_angl_out, v.t_out_avg[istru, i_rf, i_angle, i_lam], 'b--', label="out. Transmission")
                           
                        plt.ylim([1e-6,1.])
                        plt.ylabel("Efficiencies")
                        plt.yscale("log")
                        plt.legend()
                        if (params.unit_lambda_plot=='Hz'):
                            ax1 = plt.gca()
                            ax2 = ax1.twiny()
                            um_lambdas = 299792458/l_lambdas * 1e-6
                            min_int_lambdas = int(um_lambdas[0])
                            max_int_lambdas = int(um_lambdas[-1])
                            int_lambdas = np.arange(min_int_lambdas, max_int_lambdas+1)
                            ax2.set_xticks(1/int_lambdas)
                            ax2.set_xticklabels(str(int_lambdas))
                            ax1.set_xlabel("Frequency (Hz)")
                            ax2.set_xlabel("Outgoing angle")   
                        else:
                            plt.xlabel("Outgoing angle")
                        plt.tight_layout()
                        if (save):
                            plt.savefig(filename + ".svg")
                        print(filename)
                        plt.show()
                        plt.clf()
                        
                        
                        if ("refl" in kept_modes):
                            kl = kx0 - 2*np.pi/(l_structure[istru].period)
                            kr = kx0 + 2*np.pi/(l_structure[istru].period)
                            angle_l = np.arcsin(kl/k0)*180/np.pi
                            angle_r = np.arcsin(kr/k0)*180/np.pi
                            post.traitement(r_angl_out, v.r_out_avg[istru, i_rf, i_angle, i_lam], angle_l=angle_l, angle_r=angle_r, filename=file_root, doss="desordre/post_process")
    
                    else:
                        # Not averaging AND more than one repetition, so one plot
                        # per rep
    
                        for irep in range(params.nb_reps):
    
                            filename = (path + "/" + file + "_per_" + str(per)
                                        + "_prof_" + str(prof) + "_width_" + str(width)
                                        + "_RF_" + str(rf) + "_VAR_" + str(var) + "_theta_" + str(theta)
                                        + "_phi_" + str(phi) + "_psi_" + str(psi)
                                        + "_irep_" + str(irep) + "_modes_" + str(modes))
    
                            plt.figure(figsize=(10,10))
                            if ("refl" in kept_modes):
                                refl_modes = v.r_out_angl[istru, i_rf, i_angle, i_lam]
                                k0 = 2*np.pi / l_lambdas[i_lam]
                                kx0 = k0 * np.sin(l_angle[i_angle].theta)
                                kx_per = 2*np.pi/(l_structure[istru].period*params.super_period)
                                refl_k = kx0 + refl_modes*kx_per
                                r_angl_out = np.arcsin(refl_k / (2*np.pi / l_lambdas[i_lam]))*180/np.pi
    #                            print(refl_modes, kx0, kx_per, r_angl_out)
                                plt.plot(r_angl_out, v.r_out_reps[istru, i_rf, i_angle, i_lam, irep], 'b', label="out. Reflection")
                               
                            if ("tran" in kept_modes):
                                tran_modes = v.t_out_angl[istru, i_rf, i_angle, i_lam]
                                k0 = 2*np.pi / lamb
                                kx0 = k0 * np.sin(l_angle[i_angle].theta)
                                kx_per = 2*np.pi/(l_structure[istru].period*params.super_period)
                                tran_k = kx0 + tran_modes*kx_per
                                t_angl_out = np.arcsin(tran_k / 2*np.pi / lamb)*180/np.pi
                                plt.plot(t_angl_out, v.t_out_reps[istru, i_rf, i_angle, i_lam, irep], 'b--', label="out. Transmission")
                            
                              
                            plt.ylim([1e-6,1.])
                            plt.ylabel("Efficiencies")
                            plt.yscale("log")
                            if (params.unit_lambda_plot=='Hz'):
                                ax1 = plt.gca()
                                ax2 = ax1.twiny()
                                ax2.set_xlabel("Wavelength ($\mu$m)")
                                ax1.set_xlabel("Frequency (Hz)")
                                um_lambdas = 299792458/l_lambdas * 1e6
                                min_int_lambdas = int(um_lambdas[0])
                                max_int_lambdas = int(um_lambdas[-1])
                                int_lambdas = np.arange(min_int_lambdas, max_int_lambdas+1)
                                str_lambdas = [str(i) for i in int_lambdas]
                                ax2.set_xlim(ax1.get_xlim())
                                ax2.set_xticks(299792458/int_lambdas*1e6)
                                ax2.set_xticklabels(str_lambdas)
                            else:
                                plt.xlabel("Outgoing angle")
                            plt.legend()
                            plt.tight_layout()
                            if (save):
                                plt.savefig(filename + ".svg")
                            print(filename)
                            plt.show()
                            plt.clf()


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

