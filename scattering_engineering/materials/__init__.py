# -*- coding: utf-8 -*-

#from epsconst import epsconst
#from epsAu import epsAu
BDD = dict()

def load_BDD():
    global BDD
    import os, importlib

    #repertoire de BMM.materials.__init__:
    rep = os.path.dirname(__file__)
    #liste des fichiers et filtrage de ceux en "eps*.py"
    materials = [fn.replace('.py', '') for fn in os.listdir(rep)
                 if fn.startswith('eps') and fn.endswith('.py')]
    print(materials, rep)
    for mat in materials:
        BDD[mat] = getattr(importlib.import_module('scattering_engineering.materials.'+mat), mat)
        print(BDD[mat])
load_BDD()


def epsconst(materiau,lambd):
    #% lambda est defini en m
    global BDD

    import numpy as np


    if callable(materiau):          #si l'utilisateur donne une fonction
        epsilon = materiau(lambd)   #on l'evalue pour lambda

    elif isinstance(materiau, (str, np.string_)):
        try:
            epsilon = eval(materiau)
        except NameError:
            try:
                fonc_mat='eps' + materiau
                epsilon= BDD[fonc_mat](lambd)
            except KeyError:
                [mat,modele]=materiau.split('_')
                fonc_mat='eps' + mat
                epsilon= BDD[fonc_mat](lambd,modele)

    else:
        epsilon=materiau

    if not isinstance(epsilon, complex):
        epsilon=complex(epsilon)

    return epsilon


def Drude_omega(lambd,omega_p,omega_tau,eps_inf):
# Les expressions sont en cm-1
    omega=1e-2/lambd
    epsilon = eps_inf*(1-(omega_p**2)/(omega**2+1j*omega*omega_tau))
    return epsilon

def Lorentz_omega(lambd,omega_LO,omega_TO,Gamma_p,eps_inf):
    # Les expressions sont en cm-1
    omega=1e-2/lambd
    epsilon = eps_inf*(1-(omega_LO**2-omega_TO**2)/(omega_TO**2-omega**2-
1j*Gamma_p*omega))
    return epsilon

def Lorentz2_omega(lambd,Ap,omega_p,Gamma_p,eps_inf):
    # Les expressions sont en cm-1
    omega=1e-2/lambd
    epsilon =eps_inf+sum((Ap*omega_p**2)/(omega_p**2-omega**2-1j*Gamma_p*omega))

    return epsilon

def Lorentz3_omega(lambd,omega_0,omega_p,Gamma_p,eps_inf):
    # Les expressions sont en cm-1
    omega=1e-2/lambd
    epsilon =eps_inf+sum((omega_p**2)/(omega_0**2-omega**2-1j*Gamma_p*omega))

    return epsilon

def Brendel_model(lambd,f0,Gamma_0,omega_p,sigma_j,Gamma_j,omega_j,f_j,eps_inf=1,units_model='m'):
#    Brendel and Bormann, JAP (1991)
# lambd est la longueur d'onde, f0, Gamma_0,omega_p définissent la composante Drude
# sigma_j, Gamma_j, omega_j, f_j sont des arrays de même taille, paramétrisant les différents résonateurs du modèle.
    import scipy.special
    import numpy as np
    from math import pi as pi
    import BMM
    omega=BMM.conv_lambda_from_m(lambd,units_model)
#    a0j=((1+(Gamma_j/omega)**2)+1)
    a_j=(omega/np.sqrt(2))*(np.sqrt(np.sqrt(1+(Gamma_j/omega)**2)+1)+1j*np.sqrt(np.sqrt(1+(Gamma_j/omega)**2)-1))
#    if min(np.imag(a_j))<0:
#        print a_j
    Omega_p=np.sqrt(f0)*omega_p
    chi_j=(1j*np.sqrt(pi)*omega_p**2/(2*np.sqrt(2)))*(f_j/(a_j*sigma_j))*(scipy.special.wofz((a_j-omega_j)/(np.sqrt(2)*sigma_j))+scipy.special.wofz((a_j+omega_j)/(np.sqrt(2)*sigma_j)))
    epsilon=eps_inf-Omega_p**2/(omega*(omega+1j*Gamma_0))+np.sum(chi_j)

    return epsilon



def plot_epsilon(materiau,modele,tablambd,trace='epsilon',unit_lambda='m',save='non',format_save='pdf'):
# Trace l'indice ou la permittivite d'un materiau defini dans la base de donnee sur la bande spectrale tablambda
# défini avec l'unité unit_lambda
#    Option trace au choix ='nk' trace l'indice ou ='epsilon' trace la permittivité
# On pourrait tracer la conductivité, sachant que epsilon_0*epsilon=epsilon_0*epsilon_r+i*sigma/omega
    import matplotlib.pyplot as plt
    import pylab
    import BMM
    import BMM.materials
    import numpy as np
    global BDD
    epsilon=np.empty(np.shape(tablambd),dtype=complex)
    tablambda=BMM.conv_lambda(tablambd,unit_lambda)
    for numlambda,lambd in enumerate(tablambda):
        fonc_mat='eps' + materiau
        epsilon[numlambda]=BDD[fonc_mat](lambd,modele)

    if trace=='epsilon':
        plt.plot(tablambd,-(np.real(epsilon)),tablambd,(np.imag(epsilon)))
        pylab.legend((r'$-\epsilon _r$', r'$\epsilon _i$'),loc='best')
    elif trace=='eps_loglog':
        plt.loglog(tablambd,np.abs(np.real(epsilon)),tablambd,(np.imag(epsilon)))
        pylab.legend((r'$-\epsilon _r$', r'$\epsilon _i$'),loc='best')
    elif trace=='cond':
#        from math import pi
        conductivite=(1/60.)*np.imag(epsilon)/tablambda
        plt.loglog(tablambd,conductivite)
#        pylab.legend((r'$ \sigma (S/m) $'),loc='best')
        plt.xlabel("Wavelength ("+unit_lambda+")")
        plt.ylabel("Conductivity (S/m)")
        if save is 'oui':
            plt.savefig("Conductivity_"+materiau+"_"+modele+"."+format_save, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1)
    else:
        print (trace)
        nk=np.sqrt(epsilon)
        plt.plot(tablambd,np.real(nk),tablambd,np.imag(nk))
        pylab.legend((r'n', r'k'),loc='best')

#    pylab.legend((r'n', r'k'))
#    ltext = pylab.gca().get_legend().get_texts()
    plt.ion()
    plt.show()
    return epsilon
