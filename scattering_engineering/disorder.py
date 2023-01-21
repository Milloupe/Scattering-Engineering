
"""
Created on Sat Jan 21 21:59:00 2023

@author: dlangevin

NOTE : In comments, the variables which depend on the slit are called
    'slit variables',
    those which depend on a mode of the Rayleigh decomposition are called
    'Rayleigh decomposiition variables'.
    Also, l always refers to a slit and n to a kx Rayleigh mode,
"""

import numpy as np

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
