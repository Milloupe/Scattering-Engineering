
"""
Created on Thu May 16 08:38:27 2019

@author: dlangevin

NOTE : In comments, the variables which depend on the slit are called
    'slit variables',
    those which depend on a mode of the Rayleigh decomposition are called
    'Rayleigh decomposiition variables'.
    Also, l always refers to a slit and n to a kx Rayleigh mode,
"""

import numpy as np
import sys as sys
import numpy.random as rdm
from .disorder import init_fentes_sillons


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
    p.eta_2 = [p.eps_struct[l] * p.k0 / (1.0j * np.sqrt(p.eps_m)) for l in range(p.nb_fentes_tot)]

    if (len(p.ep_sub)==0):
        # No substrate layer
        p.sub = False
        p.eta_3 = p.eps_3 * p.k0 / (1.0j * np.sqrt(p.eps_m))

    elif (len(p.ep_sub) >= 1):
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
    if (len(p.ep_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    p.int = np.array(p.interf * p.super_period)
    offsets = np.array([[p.period * i] * len(p.interf) for i in range(p.super_period)])
    p.int += np.concatenate(offsets)
    p.h = np.array(p.prof * p.super_period)
    p.L = p.period * p.super_period
    p.nb_fentes_tot = len(p.int)//2

    init_fentes_sillons(p)


def init_super_low_discrep_sequence(p, alpha=(np.sqrt(5)+1)/2.):
    """
        Initialises the slit positions with a low discrepancy sequence
        based on the Irrational Additive method.
        Alpha is the irrational used for the sequence, default is 1/phi (golden ratio)

        Completely disregards the given interfaces, except for the slit widths
        (USE ONLY WITH ONE SLIT FOR THE MOMENT)
    """
    if (len(p.ep_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    width = p.interf[1] - p.interf[0] # USE ONLY WITH ONE SLIT
    new_pos = []
    new_x = p.random_factor
    p.L = p.period * p.super_period
    for i in range(p.super_period):
        new_x = np.mod(new_x+alpha, 1) # The Irrational Additive Sequence
        new_pos.append(new_x*p.L)          # Slit beg
        new_pos.append(new_x*p.L + width)    # Slit end

    p.int = np.sort(new_pos)
    if not(check_overlap(p.int, p.L)):
        print(new_pos, p.int)
        sys.exit("Low discrepancy sequence problem: The sequence chosen causes an overlap.")
    p.h = np.array(p.prof * p.super_period)
    p.nb_fentes_tot = len(p.int)//2

    init_fentes_sillons(p)


def init_super_random_correl(interf, prof, L, epaiss_metal, p):
    """
        >> If p.super_period is 1, simply initialises the period structure
            Otherwise, repeats and randomises the period a given number of times
            in order to compute the super-diffracted orders (i.e., diffusion)
    """
    if (len(p.ep_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    nbSlit = len(interf)
    if (nbSlit > 0):
        p.int = np.zeros(nbSlit*p.super_period)
        min_shift = 2e-7 - np.array(interf[::2])
        max_shift = L - np.array(interf[1::2]) - 2e-7
        for i in range(p.super_period):
            done = 0
            while(not(done)):
                rand = np.zeros(nbSlit)
                for j in range(nbSlit//2):
                    pos_shift = rdm.uniform(min_shift[j], max_shift[j])*p.random_factor
                    # Draw uniformly a shift in position for the current slit
                    rand[2*j] = pos_shift
                    rand[2*j+1] = pos_shift
                if (check_overlap(rand + interf, L)):
                    p.int[i*nbSlit:(i+1)*nbSlit] = L*i + rand + interf
                    done = 1
    else:
        p.int = np.array(interf * p.super_period)

    p.h = np.array(prof * p.super_period)
    p.L = L * p.super_period
    p.nb_fentes_tot = len(p.int)//2

    init_fentes_sillons(p)


def init_super_random_geometry(p, rand_type="all"):
    """
        >> Randomises the structure by shifting randomly the right interface
            of each structure
        >> The new positions are picked ar random on a uniform distribution
    """
    if (len(p.ep_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    nbSlit = len(p.interf)//2
    all_int = np.concatenate([np.array(p.interf) + p.period*i for i in range(p.super_period)])

    if rand_type == "all":
        to_move = range(nbSlit)
    else:
        to_move = rand_type[0]

    if (nbSlit > 0):
        p.int = np.zeros(nbSlit * 2 * p.super_period)
        min_shift = -p.random_factor / 2.0
        max_shift = p.random_factor / 2.0
        done = 0
        while(not(done)):
            shift = np.zeros(nbSlit * 2 * p.super_period)
            for i in range(p.super_period):
                for j in to_move:
                    pos_shift = rdm.uniform(min_shift, max_shift)
                    # Draw uniformly a shift in geometry for the current slit
                    shift[nbSlit*2*i + 2*j] = 0
                    shift[nbSlit*2*i + 2*j+1] = pos_shift
                    # Shifting only the right interface
            if (check_overlap(shift + all_int, p.period*p.super_period)):
                p.int = shift + all_int
                done = 1
    else:
        p.int = np.array(p.interf * p.super_period)

    p.h = np.array(p.prof * p.super_period)
    p.L = p.period * p.super_period
    p.nb_fentes_tot = len(p.int)//2

    init_fentes_sillons(p)


def init_super_jitter(p, rand_type="all"):
    """
        >> Randomises the structure where p.super_period slits lay in
            a period of p.super_period*L
            Places the slits given in interf/y at their normal position,
            then applies a jitter
        >> The jitter is picked at random on a uniform distribution of size
            p.rand_factor
        >> The rand variable determines whether all slits or only a subgroup
            should be randomised.
    """
    if (len(p.ep_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    nbSlit = len(p.interf)//2
    all_int = np.concatenate([np.array(p.interf) + p.period*i for i in range(p.super_period)])

    if rand_type == "all":
        to_move = range(nbSlit)
    else:
        to_move = rand_type[0]

    if (nbSlit > 0):
        p.int = np.zeros(nbSlit * 2 * p.super_period)
        min_shift = -p.random_factor / 2.0
        max_shift = p.random_factor / 2.0
        done = 0
        while(not(done)):
            rand = np.zeros(nbSlit * 2 * p.super_period)
            for i in range(p.super_period):
                for j in to_move:
                    pos_shift = rdm.uniform(min_shift, max_shift)
                    # Draw uniformly a shift in position for the current slit
                    rand[nbSlit*2*i + 2*j] = pos_shift
                    rand[nbSlit*2*i + 2*j+1] = pos_shift
            if (check_overlap(rand + all_int, p.period*p.super_period)):
                p.int = rand + all_int
                done = 1
    else:
        p.int = np.array(p.interf * p.super_period)

    p.h = np.array(p.prof * p.super_period)
    p.L = p.period * p.super_period
    p.nb_fentes_tot = len(p.int)//2

    init_fentes_sillons(p)


def init_super_rsa(p):
    """
        Initialises the slit positions with a RSA methode
        Exluding radius, by default, is the skin depth used in check_overlap

        Completely disregards the given interfaces, except for the slit widths
        (USE ONLY WITH ONE SLIT FOR THE MOMENT)
    """
    if (len(p.ep_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    width = p.interf[1] - p.interf[0] # USE ONLY WITH ONE SLIT
    new_pos = []
    p.L = p.period * p.super_period

    i = 0
    while (i < p.super_period):
        pos = rdm.random()*p.L
        new_pos.extend([pos, pos+width])
        if not(check_overlap(new_pos, p.L)):
            new_pos.pop()
            new_pos.pop()
        else:
            i += 1

    p.int = np.sort(new_pos)
    p.h = np.array(p.prof * p.super_period)
    p.nb_fentes_tot = len(p.int)//2

    init_fentes_sillons(p)


def check_overlap(slits, L):
    """
        >> Checks that there is no overlap in the randomised PERIOD
    """
    beg_slit = slits[::2]
    end_slit = slits[1::2]
    no_overlap = True
    nb_slits = len(slits)//2
    for i in range(nb_slits):
        if (beg_slit[i] < 0 or end_slit[i] > L-2e-7):
            # The slit is out of the original period
#            print("Rejected 1")
            return False
        pad_beg = beg_slit[i] - 2e-7
        pad_end = end_slit[i] + 2e-7
        for j in range(i):
            # We also consider that there is an overlap when the interfaces
            # are closer than 200 nm
            if (beg_slit[j] > pad_beg and beg_slit[j] < pad_end):
#                print("Rejected 2")
                return False
            elif (end_slit[j] > pad_beg and end_slit[j] < pad_end):
#                print("Rejected 3")
                return False
            elif (end_slit[i] > beg_slit[j] and end_slit[i] < end_slit[j]):
#                print("Rejected 4")
                return False
    return no_overlap


def check_jitter(slits, L):
    """
        >> Checks that there is no overlap in the randomised STRUCTURE
    """
    beg_slit = slits[::2]
    end_slit = slits[1::2]
    no_overlap = True
    nb_slits = len(slits)//2
    for i in range(nb_slits):
        pad_beg = (beg_slit[i] - 2e-7) % L
        pad_end = (end_slit[i] + 2e-7) % L
        # We also consider that there is an overlap when the interfaces
        # are closer than 200 nm
        for j in range(i):
            if (pad_end > pad_beg):
                if (beg_slit[j] > pad_beg and beg_slit[j] < pad_end):
                    print("Rejected 1.1")
                    return False
                if (end_slit[j] > pad_beg and end_slit[j] < pad_end):
                    print("Rejected 1.2")
                    return False
                if (pad_beg > beg_slit[j] and pad_end < end_slit[j]):
                    print("Rejected 1.3")
                    return False
            elif (pad_beg > pad_end):
                # The slit loops over the unit cell
                end_slit = end_slit[j] % L
                beg_slit = beg_slit[j] % L
                if (end_slit < beg_slit):
                    # The other slit loops over the unit cell too
                    print("Rejected 2.1")
                    return False
                if (end_slit > pad_beg):
                    print("Rejected 2.2")
                    return False
                if (beg_slit < pad_end):
                    print("Rejected 2.3")
                    return False
    return no_overlap


def init_gaussian_super_random(p):
    """
        >> Randomises the structure where p.super_period slits lay in
            a period of p.super_period*L
            Places the slits given in interf/y as a block and does not
            randomise their relative positions
        >> Picks iteratively within a gaussian ditribution the position of
            the next slit, centered on L and with p.random_factor
            as a standard deviation
    """
    if (len(p.ep_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    nbSlit = len(p.interf)
    if (nbSlit > 0):
        done = 0
        p.h = np.array(p.prof * p.super_period)
        p.L = p.period * p.super_period
        nbSlit = len(p.interf)
        end_int = p.interf[-1]
        while(not(done)):
            done = 1
            p.int = np.zeros(nbSlit*p.super_period)
            p.int[:nbSlit] = p.interf
            # Places the first slit at the origin
            for i in range(1, p.super_period):
                decal = rdm.normal(loc=p.L, scale=p.random_factor)
                pos = p.int[nbSlit*i-nbSlit] + max([decal, end_int+2e-7])
                p.int[nbSlit*i:nbSlit*(i+1)] = pos + np.array(p.interf)
                if (p.int[2-i+1] > p.L):
                    done = 0
                    print("too far apart!")
    else:
        p.int = []
        p.h = []
        p.L = 1.0
    p.nb_fentes_tot = len(p.int)//2

    init_fentes_sillons(p)


def init_correl_super_random(p, variance):
    """
        >> Randomises the structure where p.super_period slits lay in
            a period of p.super_period*L
            Places the slits given in interf/y as a block and does not
            randomise their relative positions
        >> Uses the fourier transform of a gaussian noise to choose
            the new positions
            Works only for single-slit arrays at the moment.
    """
    if (len(p.ep_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    all_int = np.concatenate([np.array(p.interf) + p.period*i for i in range(p.super_period)])
    nbSlit = len(all_int)//2

    if (nbSlit > 0):
        done = 0
        p.h = np.array(p.prof * p.super_period)
        p.L = p.period * p.super_period
        p.int = np.zeros(nbSlit*p.super_period)

        d = p.period*1e6
        while(not(done)):
            per_pos = all_int[::2]
            delta_pos = np.zeros_like(per_pos, dtype=float)

            L = p.random_factor*1e6 # Moving to um to normalise
            var = variance
            sigma = nbSlit*d/(2*np.pi*L)

            ns = np.arange(1, 4*sigma+1)

            phi = np.random.random(len(ns))*2*np.pi

            for i in range(nbSlit):
                to_sum = (np.exp(1j*phi - ns**2/(2*sigma**2)  + 2j*np.pi*ns*(i)/(nbSlit*d)))
                delta_pos[i] = np.real(np.sum(to_sum))

            delta_pos = delta_pos*(np.sqrt(nbSlit/np.sum(delta_pos**2)))*d*var*1e-6
            delta = np.concatenate(np.tile(delta_pos, (2,1)).T)
            # Just making sure we shift the beginning and end of each slit
            # by the same delta
            new_pos = all_int + delta

            if (check_overlap(new_pos, p.L)):
                done = 1
                p.int = new_pos
            else:
                print("Overlap")
    else:
        p.int = []
        p.h = []
        p.L = 1.0
    p.nb_fentes_tot = len(p.int)//2

    init_fentes_sillons(p)

def init_sterl_correl_super_random(p, variance):
    """
        >> Randomises the structure where p.super_period slits lay in
            a period of p.super_period*L
            Places the slits given in interf/y as a block and does not
            randomise their relative positions
        >> Uses the fourier transform of a gaussian noise to choose
            the new positions
            Works only for single-slit arrays at the moment.
    """
    if (len(p.ep_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    all_int = np.concatenate([np.array(p.interf) + p.period*i for i in range(p.super_period)])
    nbSlit = len(all_int)//2

    if (nbSlit > 0):
        done = 0
        p.h = np.array(p.prof * p.super_period)
        p.L = p.period * p.super_period
        p.int = np.zeros(nbSlit*p.super_period)

        d = p.period*1e6
        while(not(done)):
            per_pos = all_int[::2]*1e6
            delta_pos = np.zeros_like(per_pos, dtype=float)

            L = p.random_factor*1e6 # Moving to um to normalise
            var = variance

            shifts = rdm.random(nbSlit)-1/2.
            shifts = shifts*var*d

            cross_shift = np.zeros((nbSlit,nbSlit))
            for i in range(nbSlit):
                for j in range(nbSlit):
                    dist = np.abs(per_pos[i]-per_pos[j])
                    correl = np.exp(-(dist/(2*L*d))**2)
                    cross_shift[i,j] = shifts[j]*correl
                    cross_shift[j,i] = shifts[i]*correl
            tot_cross = np.sum(cross_shift, axis=1)

            # plt.plot(tot_cross)
            # plt.plot(shifts)
            delta_pos = (tot_cross + shifts)*1e-6
            delta = np.concatenate(np.tile(delta_pos, (2,1)).T)
            # Just making sure we shift the beginning and end of each slit
            # by the same delta
            new_pos = all_int + delta

            if (check_overlap(new_pos, p.L)):
                done = 1
                p.int = new_pos
            else:
                print("Overlap")
    else:
        p.int = []
        p.h = []
        p.L = 1.0
    p.nb_fentes_tot = len(p.int)//2

    init_fentes_sillons(p)


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
        sys.exit("Erreur : Nb de fentes en x et de profondeurs différents.")
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
        p.kz_sub = np.zeros((len(p.ep_sub), 2*p.nb_modes+1), dtype=complex)
        p.prop_ord_sub = [[]]
        for isub in range(1, len(p.ep_sub)):
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
            for isub in range(len(p.ep_sub)):
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
    p.eps_eta_eps1 = [p.eps_1 / (p.eta_1 * p.eps_struct[l]) for l in range(p.nb_fentes_tot)]
    if not(p.sub):
        p.eps_eta_eps3 = [p.eps_3 / (p.eta_3 * p.eps_struct[l]) for l in range(p.nb_fentes_tot)]
    else:
        p.eps_eta_epssub = [p.eps_sub[0] / (p.eta_sub * p.eps_struct[l]) for l in range(p.nb_fentes_tot)]

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

    p.kxd = np.zeros(p.nb_fentes_tot, dtype=complex)
    p.neglig = np.zeros(p.nb_fentes_tot, dtype=complex)
    for l in range(p.nb_fentes_tot):
        p.kxd[l], p.neglig[l] = calc_kxd(l, p)

    comp_help_variables(p)
    # Computing some often-used variables, to avoid repeating calculations

    p.kzd = np.array([calc_kzd(l, p) for l in range(p.nb_fentes_tot)])
    p.K = np.array([calc_K(l, p) for l in range(p.nb_fentes_tot)])

    n_modes = np.concatenate([np.arange(0, p.nb_modes+1), np.arange(-p.nb_modes,0)])
    p.Alpha_m1 = np.array([calc_Alpha_m(l, 1, p) for l in range(p.nb_fentes_tot)])
    p.Alpha_p1 = np.array([calc_Alpha_p(l, 1, p) for l in range(p.nb_fentes_tot)])

    p.Gamma1 = np.array([calc_Gamma(l, p) for l in range(p.nb_fentes_tot)])

    # Reordering the indices to have them in the correct order

    p.I1 = np.array([[calc_I(n, l, 1, p) for l in range(p.nb_fentes_tot)]
                    for n in n_modes])
    p.Int1 = np.array([[calc_Int(n, l, 1, p) for l in range(p.nb_fentes_tot)]
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
    C_sub = np.zeros((len(p.ep_sub), 2*p.nb_modes+1), dtype=complex)
    P_sub = np.zeros((len(p.ep_sub), 2*p.nb_modes+1), dtype=complex)
    for n in n_modes:
        A = p.kz_sub[-1, n]/p.eps_sub[-1] - p.kz3[n]/p.eps_3
        B = p.kz_sub[-1, n]/p.eps_sub[-1] + p.kz3[n]/p.eps_3
        C_sub[-1, n] = A / B * np.exp(2j*p.kz_sub[-1,n]*np.abs(p.ep_sub[-1]))
        P_sub[-1, n] = 2 / (1 + p.kz3[n]*p.eps_sub[-1]/(p.kz_sub[-1,n]*p.eps_3))
        for isub in range(1, len(p.ep_sub)):
             A = 1 + p.kz_sub[-isub,n]*p.eps_sub[-isub-1]/(p.kz_sub[-isub-1,n]*p.eps_sub[-isub])
             B = 1 - p.kz_sub[-isub,n]*p.eps_sub[-isub-1]/(p.kz_sub[-isub-1,n]*p.eps_sub[-isub])
             P_sub[-isub-1, n] = 2 / (A + B * C_sub[-isub, n])

             if (p.ep_sub[-isub-1] >= 0):
                 A = p.kz_sub[-isub, n]/p.eps_sub[-isub] * (2*P_sub[-isub-1,n]*C_sub[-isub,n]-1) + p.kz_sub[-isub-1, n]/p.eps_sub[-isub-1]
                 B = p.kz_sub[-isub-1, n]/p.eps_sub[-isub-1] + p.kz_sub[-isub, n]/p.eps_sub[-isub]
                 C_sub[-isub-1,n] = A / B * np.exp(2j*p.kz_sub[-isub-1,n]*p.ep_sub[-isub-1])
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

    elif (l2 < p.nb_fentes_tot):
        ll1 = p.grooves[l1]
        ll2 = p.slits[l2 - p.nb_grooves]
        res = (1.0 / p.L) * np.sum(p.I1[:, ll1] * p.Int1[:, ll2] / p.Beta_m1)
        A2 = p.Alpha_m1[ll2] * np.exp(1.0j * p.kzd[ll2] * p.ep_metal)
        return res * A2

    else:
        ll1 = p.grooves[l1]
        ll2 = p.slits[l2 - p.nb_fentes_tot]
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

        elif(l2 < p.nb_fentes_tot):
#            print("Case top centre: l1=", l1, " l2=", l2, end=" ")
            ll1 = p.slits[l1]
            ll2 = p.slits[l2 - p.nb_grooves]
#            print("ll1=", ll1, " ll2=", ll2)
            # The indices necessary to compute the correct values,
            # i.e. the one corresponding to the slit position within the
            # whole array of structures
            res = (1.0 / p.L) * np.sum(p.I1[:, ll1] * p.Int1[:, ll2] / p.Beta_m1)
            res = res * p.Alpha_m1[ll2] * np.exp(1.0j * p.kzd[ll2] * p.ep_metal)

            if (ll1 == ll2):
                res += - p.K[ll2] * np.exp(1.0j * p.kzd[ll2] * p.ep_metal)
            return res

        else:
#            print("Case top right: l1=", l1, " l2=", l2, end=" ")
            ll1 = p.slits[l1]
            ll2 = p.slits[l2 - p.nb_fentes_tot]
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

        elif(l2 < p.nb_fentes_tot):
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
            ll2 = l2 - p.nb_fentes_tot
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
                res = res * p.Alpha_m3[ll2] * np.exp(1.0j * p.kzd[p.slits[ll2]] * p.ep_metal)
            else:
                A = 1 + p.C_sub[0, :]
                B = p.Beta_msub + p.C_sub[0, :] * p.Beta_psub
                res = (1.0 / p.L) * np.sum(p.Isub[:, ll1] * p.Intsub[:, ll2] * A / B)
                res = res * p.Alpha_msub[ll2] * np.exp(1.0j * p.kzd[p.slits[ll2]] * p.ep_metal)
            if (ll1 == ll2):
                res += - p.K[p.slits[ll2]] * np.exp(1.0j * p.kzd[p.slits[ll2]] * p.ep_metal)
            return res



def vect_IA(l1, p):
    """
        >> Computes the right hand side of the linear system
    """
    if (l1 < p.nb_grooves):
        return - p.I1[0, p.grooves[l1]] * (1.0 - p.Beta_p1[0] / p.Beta_m1[0])
    elif (l1 < p.nb_fentes_tot):
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
                        for l2 in range(p.nb_fentes_tot + p.nb_slits)]
                        for l1 in range(p.nb_grooves)])
        p.MA_b = np.array([[matrix_MA_slits(l1, l2, p)
                        for l2 in range(p.nb_fentes_tot + p.nb_slits)]
                        for l1 in range(2*p.nb_slits)])
        p.MA = np.block([[p.MA_t],
                         [p.MA_b]])
        p.IA = np.array([vect_IA(l1,  p)
                        for l1 in range(p.nb_fentes_tot + p.nb_slits)])
        p.res_lin = np.linalg.solve(p.MA, p.IA)
    elif (p.nb_grooves > 0):
        p.MA = np.array([[matrix_MA_grooves(l1, l2, p)
                        for l2 in range(p.nb_grooves)]
                        for l1 in range(p.nb_grooves)])

        p.IA = np.array([vect_IA(l1,  p)
                        for l1 in range(p.nb_fentes_tot + p.nb_slits)])
        p.res_lin = np.linalg.solve(p.MA, p.IA)
    elif (p.nb_slits > 0):
        p.MA = np.array([[matrix_MA_slits(l1, l2, p)
                        for l2 in range(2*p.nb_slits)]
                        for l1 in range(2*p.nb_slits)])

        p.IA = np.array([vect_IA(l1,  p)
                        for l1 in range(p.nb_fentes_tot + p.nb_slits)])
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
    A0_3 = p.res_lin[p.nb_grooves:p.nb_fentes_tot] * np.exp(1.0j * p.kzd[p.slits] * p.ep_metal) * p.Alpha_m1[p.slits]
    B0_3 = p.res_lin[p.nb_fentes_tot:] * p.Alpha_p1[p.slits]
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
    if (p.nb_fentes_tot > 0):
        Rnmy(p)


def Tnmy(p):
    """
        Computes the y reflection coefficients for all orders
    """
    p.Tnmy3 = np.zeros(2 * p.nb_modes + 1, dtype=complex)

    if not(p.sub):
        A0_3 = p.res_lin[p.nb_grooves:p.nb_fentes_tot] * p.Alpha_p3
        B0_3 = p.res_lin[p.nb_fentes_tot:] * p.Alpha_m3 * np.exp(1.0j * p.kzd[p.slits] * p.ep_metal)
        res_3 = A0_3 + B0_3

        for n in range(-p.nb_modes, p.nb_modes+1):
            to_sum = res_3 * p.Int3[n, :]
            A = p.Beta_m3[n]
            p.Tnmy3[n] = np.sum(to_sum) / (p.L * A)
    else:
        A0_sub = p.res_lin[p.nb_grooves:p.nb_fentes_tot] * p.Alpha_psub
        B0_sub = p.res_lin[p.nb_fentes_tot:] * p.Alpha_msub * np.exp(1.0j * p.kzd[p.slits] * p.ep_metal)
        res_sub = A0_sub + B0_sub

        for n in range(-p.nb_modes, p.nb_modes+1):
            A = np.sum(res_sub * p.Intsub[n, :])
            B = (p.L * (p.Beta_msub[n] + p.Beta_psub[n]*p.C_sub[0,n]))

            C = np.exp(1.0j * np.sum(p.kz_sub[:, n] * np.abs(p.ep_sub)))
            D = np.prod(p.P_sub[:, n])
            p.Tnmy3[n] = A * C * D / B


def Tnm(p):
    """
        Computes the reflection coefficients
    """
    if (p.nb_fentes_tot > 0):
        Tnmy(p)


def reflec_orders(p):
    """
        Computes the reflectivity in each diffraction order
    """
    if (p.nb_fentes_tot > 0):
        p.ref_ord = np.real(p.kz1)/p.kz01 * np.absolute(p.Rnmy)**2


def transm_orders(p):
    """
        Computes the transmittivity in each diffraction order
    """
    if (p.nb_fentes_tot > 0):
        p.trans_ord = p.eps_1 / (p.eps_3 * p.kz01) * np.real(p.kz3) * np.absolute(p.Tnmy3)**2
