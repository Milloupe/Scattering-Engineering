
"""
Created on Thu May 16 08:38:27 2019

@author: Denis Langevin

NOTE : In comments, the variables which depend on the slit are called
    'slit variables',
    those which depend on a mode of the Rayleigh decomposition are called
    'Rayleigh decomposiition variables'.
    Also, l always refers to a slit and n to a kx Rayleigh mode,
"""

import numpy as np
import sys as sys
import numpy.random as rdm
from .disorder import init_slits_sillons


# Some init functions to fill in all the variables stored in profil

def init_base(p):
    """
        >> Initialises the A, k0, kx0 and kz0 of the incident field,
         and the eps_m, eta_1, eta_2 of the structure
        >> Requires lambd, L, Ly, metal to be defined in profil,
         as well as theta and phi
    """
    p.k0 = 2 * np.pi / p.lambd
    p.n1 = np.sqrt(p.eps_1)
    p.n3 = np.sqrt(p.eps_3)
    p.kx0 = p.k0 * np.sin(p.theta) * p.n1 # kx0 = kx0 (~Snell-Descartes)
    p.kz01 = np.sqrt(p.eps_1 * p.k0**2 - p.kx0**2)
    p.kz03 = p.kz01
    p.eta_1 = p.eps_1 * p.k0 / (1.0j * np.sqrt(p.eps_m))
    p.eta_2 = [p.eps_struct[l] * p.k0 / (1.0j * np.sqrt(p.eps_m)) for l in range(p.nb_slits_tot)]

    if (len(p.h_sub)==0):
        # No substrate layer
        p.sub = False
        p.eta_3 = p.eps_3 * p.k0 / (1.0j * np.sqrt(p.eps_m))

    elif (len(p.h_sub) >= 1):
        # Substrate layers
        p.sub = True
        p.eta_sub = p.eps_sub[0] * p.k0 / (1.0j * np.sqrt(p.eps_m))
        p.n_sub = np.array([np.sqrt(p.eps_sub[i]) for i in range(len(p.eps_sub))])
        p.kz0_sub = np.array([np.sqrt(p.eps_sub[i] * p.k0**2 - p.kx0**2)
                              for i in range(len(p.eps_sub))])



def init_super(p):
    """
        >> If p.super_period is 1, simply initialises the period structure
            Otherwise, repeats the period a given number of times in order
            to compute the super-diffracted orders (i.e., diffusion)
    """
    if (len(p.h_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    p.int = np.array(p.interf * p.super_period)
    offsets = np.array([[p.period * i] * len(p.interf) for i in range(p.super_period)])
    p.int += np.concatenate(offsets)
    p.h = np.array(p.depth * p.super_period)
    p.L = p.period * p.super_period
    p.nb_slits_tot = len(p.int)//2

    init_slits_sillons(p)



def init_structure(p):
    """
        >> Initialises the x(y)_deb, x(y)_fin, w and h of each slit.
        >> Requires interface, interface_y, profondeur and profondeur_y
           to have the correct dimensions.
           Assumes that the structure begins and ends with metal, so that
           the first of each two interface positions is the beginning of a slit
           and every other is its end.
    """
    eps = 2.2*1e-11
    if (len(p.int) % 2 != 0):
        sys.exit("Erreur : nb impair d'interfaces données selon x.")
    if (len(p.h) != len(p.int)/2):
        sys.exit("Erreur : Nb de slits en x et de profondeurs différents.")
    if (len(p.int) > 0 and p.int[-1] > (p.L*p.super_period-eps)):
        sys.exit("Erreur : interface au delà de la période en x.")
    p.x_deb = np.array(p.int[::2])
    p.x_fin = np.array(p.int[1::2])
    p.w = p.x_fin-p.x_deb


def init_Rayleigh(p):
    """
        >> Initialises the values of the k-vectors in the Rayleigh decomp.,
        i.e. kx, ky and kz for all modes n and m.
        >> Requires k0, kx0,  ky0, eps_1, nb_modes, L and Ly
        to be already defined in profil
    """
    p.kx = np.zeros(2 * p.nb_modes + 1, dtype=complex)
    p.kz1 = np.zeros(2 * p.nb_modes + 1, dtype=complex)
    p.prop_ord1 = []
    p.kz3 = np.zeros(2 * p.nb_modes + 1, dtype=complex)
    p.prop_ord3 = []
    if (p.sub):
        p.kz_sub = np.zeros((len(p.h_sub), 2*p.nb_modes+1), dtype=complex)
        p.prop_ord_sub = [[]]
        for isub in range(1, len(p.h_sub)):
            p.prop_ord_sub.append([])
    
    for n in range(-p.nb_modes, p.nb_modes+1):
        kx = p.kx0 + 2 * n * np.pi / p.L # No influence of the index (Lalanne omm 2000)
        kz1 = np.sqrt(p.eps_1 * p.k0**2 - kx**2 + 0j)
        kz3 = np.sqrt(p.eps_3 * p.k0**2 - kx**2 + 0j)
        p.kx[n] = kx
        p.kz1[n] = kz1
        if (np.real(kz1) != 0):
            p.prop_ord1.append(n)
        p.kz3[n] = kz3
        if (np.real(kz3) != 0):
            p.prop_ord3.append(n)
        if (p.sub):
            for isub in range(len(p.h_sub)):
                kz_sub = np.sqrt(p.eps_sub[isub] * p.k0**2 - kx**2 + 0j)
                p.kz_sub[isub, n] = kz_sub
                if (np.real(kz_sub) != 0):
                    p.prop_ord_sub[isub].append(n)


def comp_help_variables(p):
    """
        >> Computes many variables which are needed in other functions,
            in order to optimize calculations
    """
    p.half_kd_wd = p.kxd * p.w/2
    p.eps_eta_eps1 = [p.eps_1 / (p.eta_1 * p.eps_struct[l]) for l in range(p.nb_slits_tot)]
    if not(p.sub):
        p.eps_eta_eps3 = [p.eps_3 / (p.eta_3 * p.eps_struct[l]) for l in range(p.nb_slits_tot)]
    else:
        p.eps_eta_epssub = [p.eps_sub[0] / (p.eta_sub * p.eps_struct[l]) for l in range(p.nb_slits_tot)]

def init_variables(p):
    """
        >> Initialises the values of the variables used in the system,
        i.e. the integrals I, Int, K
        the slit variables Alpha(+/-), Gamma, kxd
        and the Rayleigh decomposition variables Beta(+/-).
        >> Requires L, kx0, kx(n), w(l), xdeb(l), xfin(l)
        as well as k0, eta_1, eta_2 and eps_2 to be defined in profil.
           Also, the order in which the new variables are computed is important
           (for instance, kxd must be computed before all the others,
           as it is necessary in the following functions)
    """

    p.kxd = np.zeros(p.nb_slits_tot, dtype=complex)
    p.neglig = np.zeros(p.nb_slits_tot, dtype=complex)
    for l in range(p.nb_slits_tot):
        p.kxd[l], p.neglig[l] = calc_kxd(l, p)

    comp_help_variables(p)
    # Computing some often-used variables, to avoid repeating calculations

    p.kzd = np.array([calc_kzd(l, p) for l in range(p.nb_slits_tot)])
    p.K = np.array([calc_K(l, p) for l in range(p.nb_slits_tot)])

    n_modes = np.concatenate([np.arange(0, p.nb_modes+1), np.arange(-p.nb_modes,0)])
    p.Alpha_m1 = np.array([calc_Alpha_m(l, 1, p) for l in range(p.nb_slits_tot)])
    p.Alpha_p1 = np.array([calc_Alpha_p(l, 1, p) for l in range(p.nb_slits_tot)])

    p.Gamma1 = np.array([calc_Gamma(l, p) for l in range(p.nb_slits_tot)])

    # Reordering the indices to have them in the correct order

    p.I1 = np.array([[calc_I(n, l, 1, p) for l in range(p.nb_slits_tot)]
                    for n in n_modes])
    p.Int1 = np.array([[calc_Int(n, l, 1, p) for l in range(p.nb_slits_tot)]
                    for n in n_modes])

    p.Beta_m1 = np.array([calc_Beta_m(n, 1, p) for n in n_modes])
    p.Beta_p1 = np.array([calc_Beta_p(n, 1, p) for n in n_modes])

    if not(p.sub):
        p.Alpha_m3 = np.array([calc_Alpha_m(l, 3, p) for l in range(p.nb_slits)])
        p.Alpha_p3 = np.array([calc_Alpha_p(l, 3, p) for l in range(p.nb_slits)])
        p.I3 = np.array([[calc_I(n, l, 3, p) for l in range(p.nb_slits)]
                        for n in n_modes])
        p.Int3 = np.array([[calc_Int(n, l, 3, p) for l in range(p.nb_slits)]
                        for n in n_modes])
        p.Beta_m3 = np.array([calc_Beta_m(n, 3, p) for n in n_modes])
        p.Beta_p3 = np.array([calc_Beta_p(n, 3, p) for n in n_modes])
    else:
        p.Alpha_msub = np.array([calc_Alpha_m(l, "sub", p) for l in range(p.nb_slits)])
        p.Alpha_psub = np.array([calc_Alpha_p(l, "sub", p) for l in range(p.nb_slits)])
        p.Isub = np.array([[calc_I(n, l, "sub", p) for l in range(p.nb_slits)]
                        for n in n_modes])
        p.Intsub = np.array([[calc_Int(n, l, "sub", p) for l in range(p.nb_slits)]
                        for n in n_modes])
        p.Beta_msub = np.array([calc_Beta_m(n, "sub", p) for n in n_modes])
        p.Beta_psub = np.array([calc_Beta_p(n, "sub", p) for n in n_modes])

        p.C_sub, p.P_sub = calc_sub_var(n_modes, p)
        # Not parallel for the moment because we need to compute both
        # type of variables simultaneously




# Many functions computing necessary variables


def calc_sub_var(n_modes, p):
    """n_modes,
        >> Computes the propagation variables Cj(m) and Pjj-1(m)
        >> We have the following relations:
            Rm(j) = Tm(j) x Cj(m)
            Tm(j) = Tm(j-1) x e(ikz(j-1) h(j-1)) Pjj-1(m)
    """
    C_sub = np.zeros((len(p.h_sub), 2*p.nb_modes+1), dtype=complex)
    P_sub = np.zeros((len(p.h_sub), 2*p.nb_modes+1), dtype=complex)
    for n in n_modes:
        A = p.kz_sub[-1, n]/p.eps_sub[-1] - p.kz3[n]/p.eps_3
        B = p.kz_sub[-1, n]/p.eps_sub[-1] + p.kz3[n]/p.eps_3
        C_sub[-1, n] = A / B * np.exp(2j*p.kz_sub[-1,n]*np.abs(p.h_sub[-1]))
        P_sub[-1, n] = 2 / (1 + p.kz3[n]*p.eps_sub[-1]/(p.kz_sub[-1,n]*p.eps_3))
        for isub in range(1, len(p.h_sub)):
             A = 1 + p.kz_sub[-isub,n]*p.eps_sub[-isub-1]/(p.kz_sub[-isub-1,n]*p.eps_sub[-isub])
             B = 1 - p.kz_sub[-isub,n]*p.eps_sub[-isub-1]/(p.kz_sub[-isub-1,n]*p.eps_sub[-isub])
             P_sub[-isub-1, n] = 2 / (A + B * C_sub[-isub, n])

             if (p.h_sub[-isub-1] >= 0):
                 A = p.kz_sub[-isub, n]/p.eps_sub[-isub] * (2*P_sub[-isub-1,n]*C_sub[-isub,n]-1) + p.kz_sub[-isub-1, n]/p.eps_sub[-isub-1]
                 B = p.kz_sub[-isub-1, n]/p.eps_sub[-isub-1] + p.kz_sub[-isub, n]/p.eps_sub[-isub]
                 C_sub[-isub-1,n] = A / B * np.exp(2j*p.kz_sub[-isub-1,n]*p.h_sub[-isub-1])
             else:
                 # Substrate layer, where we don't want reflection to happen
                 # -> R = 0 => C = 0
                 C_sub[-isub-1,n] = 0

    return (C_sub, P_sub)


def calc_I(n, l, zone, p):
    """
        >> Computes the slit integral I(n,l).
        >> Requires xdeb(l), xfin(l) and kx(n) to be defined in profil
    """
    if (n == 0 and p.kx0 == 0):
        return p.w[l]
    if (zone == 1):
        exp_deb = np.exp(1.0j * p.kx[n] * p.x_deb[l])
        exp_fin = np.exp(1.0j * p.kx[n] * p.x_fin[l])
        return (exp_fin - exp_deb) / (1.0j * p.kx[n])
    elif (zone == 3 or zone == "sub"):
        ll = p.slits[l]
        exp_deb = np.exp(1.0j * p.kx[n] * p.x_deb[ll])
        exp_fin = np.exp(1.0j * p.kx[n] * p.x_fin[ll])
        return (exp_fin - exp_deb) / (1.0j * p.kx[n])


def calc_Int(n, l, zone, p):
    """
        >> Computes the slit integral Int(n,l).
        >> Requires xdeb(l), xfin(l), kxd(l) and kx(n) to be defined in profil
    """
    if (n == 0 and p.kx0 == 0):
        return p.K[l]
    if (zone == 1):
        k_diff = 1/(2.0j * (p.kxd[l] - p.kx[n]))
        k_sum = 1/(2.0j * (p.kxd[l] + p.kx[n]))
        A1 = np.exp(1.0j*(p.half_kd_wd[l] - p.kx[n] * p.x_fin[l]))
        A2 = np.exp(1.0j*(-p.half_kd_wd[l] - p.kx[n] * p.x_deb[l]))
        A3 = np.exp(1.0j*(p.half_kd_wd[l] - p.kx[n] * p.x_deb[l]))
        A4 = np.exp(1.0j*(-p.half_kd_wd[l] - p.kx[n] * p.x_fin[l]))
        return k_diff * (A1 - A2) + k_sum * (A3 - A4)
    elif (zone == 3 or zone == "sub"):
        ll = p.slits[l]
        k_diff = 1/(2.0j * (p.kxd[ll] - p.kx[n]))
        k_sum = 1/(2.0j * (p.kxd[ll] + p.kx[n]))
        A1 = np.exp(1.0j*(p.half_kd_wd[ll] - p.kx[n] * p.x_fin[ll]))
        A2 = np.exp(1.0j*(-p.half_kd_wd[ll] - p.kx[n] * p.x_deb[ll]))
        A3 = np.exp(1.0j*(p.half_kd_wd[ll] - p.kx[n] * p.x_deb[ll]))
        A4 = np.exp(1.0j*(-p.half_kd_wd[ll] - p.kx[n] * p.x_fin[ll]))
        return k_diff * (A1 - A2) + k_sum * (A3 - A4)



def calc_K(l, p):
    """
        >> Computes the slit integral K(l).
        >> Requires xdeb(l), xfin(l) and kxd(l) to be defined in profil
    """
    return 2 * np.sin(p.half_kd_wd[l]) / p.kxd[l]


def calc_J(n, zone, p):
    """
        >> Computes the dielectric slab integral Jp(n).
        >> Requires L and kJ(n) to be defined in profil
    """
    if (zone == 4):
        if (p.kx[n] == 0):
            return p.L
        return (np.exp(1.0j * p.kx[n] * p.L) - 1) / (1.0j * p.kx[n])
    if (zone == 3):
        if (p.kx[n] == 0):
            return p.L
        return (np.exp(1.0j * p.kx[n] * p.L) - 1) / (1.0j * p.kx[n])


def calc_Alpha_m(l, zone, p):
    """
        >> Computes the slit variable alphax_m(l).
        >> Requires eta_1, eta_2, eps_2 and kzd(l) to be defined in profil
    """
    if (zone == 1):
        return 1 - 1.0j * p.kzd[l] * p.eps_eta_eps1[l]
    elif (zone == 3):
        return 1 - 1.0j * p.kzd[l] * p.eps_eta_eps3[l]
    elif (zone == "sub"):
        return 1 - 1.0j * p.kzd[l] * p.eps_eta_epssub[l]


def calc_Alpha_p(l, zone, p):
    """
        >> Computes the slit variable alphax_p(l).
        >> Requires eta_1, eta_2, eps_2 and kzd(l) to be defined in profil
    """
    if (zone == 1):
        return 1 + 1.0j * p.kzd[l] * p.eps_eta_eps1[l]
    elif (zone == 3):
        return 1 + 1.0j * p.kzd[l] * p.eps_eta_eps3[l]
    elif (zone == "sub"):
        return 1 + 1.0j * p.kzd[l] * p.eps_eta_epssub[l]


def calc_Gamma(l, p):
    """
        >> Computes the slit variable Gamma(l).
        >> Requires eta_2 and kzd(l) to be defined in profil
    """
    return (1.0j * p.kzd[l] - p.eta_2[l]) / (1.0j * p.kzd[l] + p.eta_2[l])


def calc_kxd(l, p):
    """
        >> Computes the slit k-vector kxd(l).
        >> Requires w(l), eps_m, eps_2 and k0 to be defined in profil
    """
    A = np.sqrt(-2.0j * p.eps_struct[l] * p.k0 / (p.w[l] * np.sqrt(p.eps_m)))
    B = np.abs((A*p.w[l]*p.eps_m )**2 / (2*p.eps_struct[l])**2)
    return A, B


def calc_kzd(l, p):
    """
        >> Computes the slit k-vector kzd(l).
        >> Requires eps_2, kxd(l), ky0 and k0 to be defined in profil
    """
    return np.sqrt(p.eps_struct[l] * p.k0**2 - p.kxd[l]**2)


def calc_Beta_m(n, zone, p):
    """
        >> Computes the Rayleigh decomposition variable Beta_m
        >> Requires eta_1 and kz(n) to be defined in profil.
    """
    if (zone == 1):
        return 1 - 1.0j * p.kz1[n] / p.eta_1
    elif (zone == 3):
        return 1 - 1.0j * p.kz3[n] / p.eta_3
    elif (zone == "sub"):
        return 1 - 1.0j * p.kz_sub[0,n] / p.eta_sub


def calc_Beta_p(n, zone, p):
    """
        >> Computes the Rayleigh decomposition variable Beta_p
        >> Requires eta_1 and kz(n) to be defined in profil.
    """
    if (zone == 1):
        return 1 + 1.0j * p.kz1[n] / p.eta_1
    elif (zone == 3):
        return 1 + 1.0j * p.kz3[n] / p.eta_3
    elif (zone == "sub"):
        return 1 + 1.0j * p.kz_sub[0,n] / p.eta_sub


def calc_Sigma(n, p):
    """
        >> Computes the dielectric slab variable sigma
        >> Requires eps_3, eps_4, kz1(n) and kz4(n) to be defined in profil.
    """
    a = p.kz3[n] * p.eps_4 - p.kz4[n] * p.eps_3
    b = p.kz3[n] * p.eps_4 + p.kz4[n] * p.eps_3
    return a / b

########
#### Computing the main matrices
########

def matrix_MA_grooves(l1, l2, p):
    """
        >> Computes and initialises the first block of the matrix describing
          the linear system, i.e. the block relative to the grooves
        >> l1 and l2 should always be less than the number of grooves
    """
    if (l2 < p.nb_grooves):
        ll1 = p.grooves[l1]
        ll2 = p.grooves[l2]
        # The indices necessary to compute the correct values,
        # i.e. the one corresponding to grooves only
        res = (1.0 / p.L) * np.sum(p.I1[:, ll1] * p.Int1[:, ll2] / p.Beta_m1)
        A2 = p.Alpha_m1[ll2] * np.exp(1.0j * p.kzd[ll2] * p.h[ll2])
        B2 = p.Gamma1[ll2] * p.Alpha_p1[ll2] * np.exp(-1.0j * p.kzd[ll2] * p.h[ll2])
        res = res * (A2 + B2)

        if (l1 == l2): # equivalent to ll1 == ll2
            A1 = np.exp(1.0j * p.kzd[ll2] * p.h[ll2])
            B1 = p.Gamma1[ll2] * np.exp(-1.0j * p.kzd[ll2] * p.h[ll2])
            res += -p.K[ll2] * (A1 + B1)
        return res

    elif (l2 < p.nb_slits_tot):
        ll1 = p.grooves[l1]
        ll2 = p.slits[l2 - p.nb_grooves]
        res = (1.0 / p.L) * np.sum(p.I1[:, ll1] * p.Int1[:, ll2] / p.Beta_m1)
        A2 = p.Alpha_m1[ll2] * np.exp(1.0j * p.kzd[ll2] * p.h_metal)
        return res * A2

    else:
        ll1 = p.grooves[l1]
        ll2 = p.slits[l2 - p.nb_slits_tot]
        res = (1.0 / p.L) * np.sum(p.I1[:, ll1] * p.Int1[:, ll2] / p.Beta_m1)
        A2 = p.Alpha_p1[ll2]
        return res * A2



def matrix_MA_slits(l1, l2, p):
    """
        >> Computes and initialises the second block of the matrix describing
          the linear system, i.e. the block relative to the slits
        >> There are 6 separate blocks, in 2 rows of 3, with only the lower
            left being 0
    """
    if (l1 < p.nb_slits):
        if(l2 < p.nb_grooves):
#            print("Case top left: l1=", l1, " l2=", l2, end=" ")
            ll1 = p.slits[l1]
            ll2 = p.grooves[l2]
#            print("ll1=", ll1, " ll2=", ll2)
            # The indices necessary to compute the correct values,
            # i.e. the one corresponding to the slit position within the
            # whole array of structures
            res = (1.0 / p.L) * np.sum(p.I1[:, ll1] * p.Int1[:, ll2] / p.Beta_m1)
            A2 = p.Alpha_m1[ll2] * np.exp(1.0j * p.kzd[ll2] * p.h[ll2])
            B2 = p.Gamma1[ll2] * p.Alpha_p1[ll2] * np.exp(-1.0j * p.kzd[ll2] * p.h[ll2])
            return res * (A2 + B2)

        elif(l2 < p.nb_slits_tot):
#            print("Case top centre: l1=", l1, " l2=", l2, end=" ")
            ll1 = p.slits[l1]
            ll2 = p.slits[l2 - p.nb_grooves]
#            print("ll1=", ll1, " ll2=", ll2)
            # The indices necessary to compute the correct values,
            # i.e. the one corresponding to the slit position within the
            # whole array of structures
            res = (1.0 / p.L) * np.sum(p.I1[:, ll1] * p.Int1[:, ll2] / p.Beta_m1)
            res = res * p.Alpha_m1[ll2] * np.exp(1.0j * p.kzd[ll2] * p.h_metal)

            if (ll1 == ll2):
                res += - p.K[ll2] * np.exp(1.0j * p.kzd[ll2] * p.h_metal)
            return res

        else:
#            print("Case top right: l1=", l1, " l2=", l2, end=" ")
            ll1 = p.slits[l1]
            ll2 = p.slits[l2 - p.nb_slits_tot]
#            print("ll1=", ll1, " ll2=", ll2)
            # The indices necessary to compute the correct values,
            # i.e. the one corresponding to the slit position within the
            # whole array of structures
            res = (1.0 / p.L) * np.sum(p.I1[:, ll1] * p.Int1[:, ll2] / p.Beta_m1)
            res = res * p.Alpha_p1[ll2]
            if (ll1 == ll2):
                res += - p.K[ll2]
            return res

    else:
        if(l2 < p.nb_grooves):
#            print("Case bot left: l1=", l1, " l2=", l2)
            return 0.0

        elif(l2 < p.nb_slits_tot):
#            print("Case bot centre: l1=", l1, " l2=", l2, end=" ")
            ll1 = l1 - p.nb_slits
            ll2 = l2 - p.nb_grooves
#            print("ll1=", ll1, " ll2=", ll2)
            # The indices necessary to compute the correct values,
            # i.e. the one corresponding to the slit position within the
            # only array of slits
            # THIS form is simpler than the other cases because most variables
            # used here are for slits only, so there is not the strange
            # shift most variables have.
            # Do note that for K and kzd, we do take the shift into account
            if not(p.sub):
                res = (1.0 / p.L) * np.sum(p.I3[:, ll1] * p.Int3[:, ll2] / p.Beta_m3)
                res = res * p.Alpha_p3[ll2]
            else:
                A = 1 + p.C_sub[0, :]
                B = p.Beta_msub + p.C_sub[0, :] * p.Beta_psub
                res = (1.0 / p.L) * np.sum(p.Isub[:, ll1] * p.Intsub[:, ll2] * A / B)
                res = res * p.Alpha_psub[ll2]
            if (ll1 == ll2):
                res += - p.K[p.slits[ll2]]
            return res

        else:
#            print("Case bot right: l1=", l1, " l2=", l2, end=" ")
            ll1 = l1 - p.nb_slits
            ll2 = l2 - p.nb_slits_tot
#            print("ll1=", ll1, " ll2=", ll2)
            # The indices necessary to compute the correct values,
            # i.e. the one corresponding to the slit position within the
            # only array of slits
            # THIS form is simpler than the other cases because most variables
            # used here are for slits only, so there is not the strange
            # shift most variables have between groove- and slit-variables.
            # Do note that for K and kzd, we take the shift into account
            if not(p.sub):
                res = (1.0 / p.L) * np.sum(p.I3[:, ll1] * p.Int3[:, ll2] / p.Beta_m3)
                res = res * p.Alpha_m3[ll2] * np.exp(1.0j * p.kzd[p.slits[ll2]] * p.h_metal)
            else:
                A = 1 + p.C_sub[0, :]
                B = p.Beta_msub + p.C_sub[0, :] * p.Beta_psub
                res = (1.0 / p.L) * np.sum(p.Isub[:, ll1] * p.Intsub[:, ll2] * A / B)
                res = res * p.Alpha_msub[ll2] * np.exp(1.0j * p.kzd[p.slits[ll2]] * p.h_metal)
            if (ll1 == ll2):
                res += - p.K[p.slits[ll2]] * np.exp(1.0j * p.kzd[p.slits[ll2]] * p.h_metal)
            return res



def vect_IA(l1, p):
    """
        >> Computes the right hand side of the linear system
    """
    if (l1 < p.nb_grooves):
        return - p.I1[0, p.grooves[l1]] * (1.0 - p.Beta_p1[0] / p.Beta_m1[0])
    elif (l1 < p.nb_slits_tot):
        return - p.I1[0, p.slits[l1-p.nb_grooves]] * (1.0 - p.Beta_p1[0] / p.Beta_m1[0])
    else:
        return 0.0



# Solving the whole problem


def resolution(p):
    """
        Solves the linear system after initialising the matrices that describe it
    """
    if (p.nb_slits > 0 and p.nb_grooves > 0):
        p.MA_t = np.array([[matrix_MA_grooves(l1, l2, p)
                        for l2 in range(p.nb_slits_tot + p.nb_slits)]
                        for l1 in range(p.nb_grooves)])
        p.MA_b = np.array([[matrix_MA_slits(l1, l2, p)
                        for l2 in range(p.nb_slits_tot + p.nb_slits)]
                        for l1 in range(2*p.nb_slits)])
        p.MA = np.block([[p.MA_t],
                         [p.MA_b]])
        p.IA = np.array([vect_IA(l1,  p)
                        for l1 in range(p.nb_slits_tot + p.nb_slits)])
        p.res_lin = np.linalg.solve(p.MA, p.IA)
    elif (p.nb_grooves > 0):
        p.MA = np.array([[matrix_MA_grooves(l1, l2, p)
                        for l2 in range(p.nb_grooves)]
                        for l1 in range(p.nb_grooves)])

        p.IA = np.array([vect_IA(l1,  p)
                        for l1 in range(p.nb_slits_tot + p.nb_slits)])
        p.res_lin = np.linalg.solve(p.MA, p.IA)
    elif (p.nb_slits > 0):
        p.MA = np.array([[matrix_MA_slits(l1, l2, p)
                        for l2 in range(2*p.nb_slits)]
                        for l1 in range(2*p.nb_slits)])

        p.IA = np.array([vect_IA(l1,  p)
                        for l1 in range(p.nb_slits_tot + p.nb_slits)])
        p.res_lin = np.linalg.solve(p.MA, p.IA)
    else:
        print("""You should never reach this case, otherwise
            you are computing the optical response of a uniform medium,
            and this is not what this code was meant for.""")


def Rnmy(p):
    """
        Computes the y reflection coefficients for all orders
    """
    p.Rnmy = np.zeros(2 * p.nb_modes + 1, dtype=complex)

    Ax = p.Alpha_m1[p.grooves] * np.exp(1.0j * p.kzd[p.grooves] * p.h[p.grooves] )
    Bx = p.Alpha_p1[p.grooves] * p.Gamma1[p.grooves] * np.exp(-1.0j * p.kzd[p.grooves] * p.h[p.grooves])

    A0_1 = p.res_lin[:p.nb_grooves] * (Ax + Bx)
    A0_3 = p.res_lin[p.nb_grooves:p.nb_slits_tot] * np.exp(1.0j * p.kzd[p.slits] * p.h_metal) * p.Alpha_m1[p.slits]
    B0_3 = p.res_lin[p.nb_slits_tot:] * p.Alpha_p1[p.slits]
    res_3 = A0_3 + B0_3

    for n in range(-p.nb_modes, p.nb_modes+1):
        to_sum1 = A0_1 * p.Int1[n, p.grooves]
        to_sum3 = res_3 * p.Int1[n, p.slits]
        p.Rnmy[n] = (np.sum(to_sum1) + np.sum(to_sum3)) / (p.L * p.Beta_m1[n])
    p.Rnmy[0] -=  p.Beta_p1[0] / p.Beta_m1[0]


def Rnm(p):
    """
        Computes the reflection coefficients
    """
    if (p.nb_slits_tot > 0):
        Rnmy(p)


def Tnmy(p):
    """
        Computes the y reflection coefficients for all orders
    """
    p.Tnmy3 = np.zeros(2 * p.nb_modes + 1, dtype=complex)

    if not(p.sub):
        A0_3 = p.res_lin[p.nb_grooves:p.nb_slits_tot] * p.Alpha_p3
        B0_3 = p.res_lin[p.nb_slits_tot:] * p.Alpha_m3 * np.exp(1.0j * p.kzd[p.slits] * p.h_metal)
        res_3 = A0_3 + B0_3

        for n in range(-p.nb_modes, p.nb_modes+1):
            to_sum = res_3 * p.Int3[n, :]
            A = p.Beta_m3[n]
            p.Tnmy3[n] = np.sum(to_sum) / (p.L * A)
    else:
        A0_sub = p.res_lin[p.nb_grooves:p.nb_slits_tot] * p.Alpha_psub
        B0_sub = p.res_lin[p.nb_slits_tot:] * p.Alpha_msub * np.exp(1.0j * p.kzd[p.slits] * p.h_metal)
        res_sub = A0_sub + B0_sub

        for n in range(-p.nb_modes, p.nb_modes+1):
            A = np.sum(res_sub * p.Intsub[n, :])
            B = (p.L * (p.Beta_msub[n] + p.Beta_psub[n]*p.C_sub[0,n]))

            C = np.exp(1.0j * np.sum(p.kz_sub[:, n] * np.abs(p.h_sub)))
            D = np.prod(p.P_sub[:, n])
            p.Tnmy3[n] = A * C * D / B


def Tnm(p):
    """
        Computes the reflection coefficients
    """
    if (p.nb_slits_tot > 0):
        Tnmy(p)


def reflec_orders(p):
    """
        Computes the reflectivity in each diffraction order
    """
    if (p.nb_slits_tot > 0):
        p.ref_ord = np.real(p.kz1)/p.kz01 * np.absolute(p.Rnmy)**2


def transm_orders(p):
    """
        Computes the transmittivity in each diffraction order
    """
    if (p.nb_slits_tot > 0):
        p.trans_ord = p.eps_1 / (p.eps_3 * p.kz01) * np.real(p.kz3) * np.absolute(p.Tnmy3)**2
