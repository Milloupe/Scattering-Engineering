
"""
Created on Sat Jan 21 21:59:00 2023

@author: Denis Langevin

NOTE : In comments, the variables which depend on the slit are called
    'slit variables',
    those which depend on a mode of the Rayleigh decomposition are called
    'Rayleigh decomposition variables'.
    Also, l always refers to a slit and n to a kx Rayleigh mode,
"""

import numpy as np
import numpy.random as rdm


def init_slits_sillons(p):
    """
        >> Initialises the array containing the indices of the slits
         that go through the surface
        >> Requires int/y, h/y, h_metal/y to be defined in profil
    """
    p.slits = []
    p.grooves = []
    for i in range(len(p.h)):
        if (p.h[i] > p.h_metal):
            if (i < len(p.h)//p.super_period):
                print("Warning: slit {} deeper than the overall width,".format(i)
                + " forced to max depth")
            p.h[i] = p.h_metal
            p.slits.append(i)
        elif (p.h[i] >= p.h_metal - 2e-7):
            p.h[i] = p.h_metal
            p.slits.append(i)
        else:
            p.grooves.append(i)
    p.nb_slits = len(p.slits)
    p.nb_grooves = p.nb_slits_tot - p.nb_slits

    if type(p.eps_2) is float:
        p.eps_struct = [p.eps_2 for l in range(p.nb_slits_tot)]
    elif (len(p.eps_2) == 1):
        p.eps_struct = [p.eps_2[0] for l in range(p.nb_slits_tot)]
    elif (len(p.eps_2) == p.nb_slits_tot):
        p.eps_struct = p.eps_2
    else:
        sys.exit("Provide either one permittivity per structure or only one.")


def init_super_low_discrep_sequence(p, alpha=(np.sqrt(5)+1)/2.):
    """
        Initialises the slit positions with a low discrepancy sequence
        based on the Irrational Additive method.
        Alpha is the irrational used for the sequence, default is 1/phi (golden ratio)

        Completely disregards the given interfaces, except for the slit widths
        (USE ONLY WITH ONE SLIT FOR THE MOMENT)
    """
    if (len(p.h_sub) != len(p.eps_sub)):
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
    p.h = np.array(p.depth * p.super_period)
    p.nb_slits_tot = len(p.int)//2

    init_slits_sillons(p)


def init_super_random_correl(interf, depth, L, epaiss_metal, p):
    """
        >> If p.super_period is 1, simply initialises the period structure
            Otherwise, repeats and randomises the period a given number of times
            in order to compute the super-diffracted orders (i.e., diffusion)
    """
    if (len(p.h_sub) != len(p.eps_sub)):
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
                    print("Overlap")
    else:
        p.int = np.array(interf * p.super_period)

    p.h = np.array(depth * p.super_period)
    p.L = L * p.super_period
    p.nb_slits_tot = len(p.int)//2

    init_slits_sillons(p)


def init_super_random_geometry(p, rand_type="all"):
    """
        >> Randomises the structure by shifting randomly the right interface
            of each structure
        >> The new positions are picked ar random on a uniform distribution
    """
    if (len(p.h_sub) != len(p.eps_sub)):
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
                print("Overlap")
    else:
        p.int = np.array(p.interf * p.super_period)

    p.h = np.array(p.depth * p.super_period)
    p.L = p.period * p.super_period
    p.nb_slits_tot = len(p.int)//2

    init_slits_sillons(p)


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
    if (len(p.h_sub) != len(p.eps_sub)):
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
                print("Overlap")
    else:
        p.int = np.array(p.interf * p.super_period)

    p.h = np.array(p.depth * p.super_period)
    p.L = p.period * p.super_period
    p.nb_slits_tot = len(p.int)//2

    init_slits_sillons(p)


def init_super_rsa(p):
    """
        Initialises the slit positions with a RSA methode
        Exluding radius, by default, is the skin depth used in check_overlap

        Completely disregards the given interfaces, except for the slit widths
        (USE ONLY WITH ONE SLIT FOR THE MOMENT)
    """
    if (len(p.h_sub) != len(p.eps_sub)):
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
            print("Overlap")
        else:
            i += 1

    p.int = np.sort(new_pos)
    p.h = np.array(p.depth * p.super_period)
    p.nb_slits_tot = len(p.int)//2

    init_slits_sillons(p)


def check_overlap(slits, L):
    """
        >> Checks that there is no overlap in the randomised PERIOD
    """
    i_slit = np.argsort(slits[::2])
    beg_slit = slits[::2][i_slit]
    end_slit = slits[1::2][i_slit]

    if (np.sum(beg_slit > end_slit)):
        print("Unexpected situation: some slits end before they begin!")

    nb_slits = len(slits)//2

    if (beg_slit[0] < 0 or end_slit[-1] > L-2e-7):
        # A slit is out of the original period
        return False

    overlap = np.sum((beg_slit[1:] - end_slit[:-1]) < 2e-7)
    # 2e-7 is a small padding to make sure we never have direct coupling
    # between slits
    return not(overlap)


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
    if (len(p.h_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    nbSlit = len(p.interf)
    if (nbSlit > 0):
        done = 0
        p.h = np.array(p.depth * p.super_period)
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
    p.nb_slits_tot = len(p.int)//2

    init_slits_sillons(p)


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
    if (len(p.h_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    all_int = np.concatenate([np.array(p.interf) + p.period*i for i in range(p.super_period)])
    nbSlit = len(all_int)//2

    if (nbSlit > 0):
        done = 0
        p.h = np.array(p.depth * p.super_period)
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
    p.nb_slits_tot = len(p.int)//2

    init_slits_sillons(p)

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
    if (len(p.h_sub) != len(p.eps_sub)):
        sys.exit("Please give one permittivity and one width per substrate layer")
    all_int = np.concatenate([np.array(p.interf) + p.period*i for i in range(p.super_period)])
    nbSlit = len(all_int)//2

    if (nbSlit > 0):
        done = 0
        p.h = np.array(p.depth * p.super_period)
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
    p.nb_slits_tot = len(p.int)//2

    init_slits_sillons(p)
