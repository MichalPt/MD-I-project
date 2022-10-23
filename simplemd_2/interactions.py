import numpy as np
import numba as nb

__all__ = [
    'interactions_zero',
    'interactions_LJ_simple',
    'interactions_zero_numba',
    'interactions_LJ_numba',
    'interactions_LJ_numpy_open',
    'interactions_LJ_numpy'
]


def interactions_zero(x, box):
    """Zero energy and forces, box is ignored."""

    return 0.0, np.zeros_like(x)


def interactions_LJ_simple(x, box, eps=1.0, sigma=1.0):
    """Simple Lennard-Jones interactions, no periodicity."""

    # number of atoms
    N = x.shape[0]

    # prepare accumulators
    U = 0.0
    dU_dx = np.zeros_like(x)

    # loop over all atoms
    for i in range(N):
        
        # loop over all possible partner atoms
        for j in range(N):

            # skip "self pair"
            if j == i:
                continue

            # displacement vector from i to j
            xij = x[j, :] - x[i, :]

            # distance squared
            d2 = (xij**2).sum()

            # evaluate pair potential and its derivative (divided by distance)
            dinv2 = 1.0 / d2
            sdinv6 = (sigma**2 * dinv2)**3
            sdinv12 = sdinv6**2
            u = 4.0*eps * (sdinv12 - sdinv6)
            du_ddd = 24.0*eps * (2*sdinv12 - sdinv6) * dinv2

            # accumulate potential energy contribution from this pair
            U += u

            # accumulate force contribution on i
            dU_dx[i, :] += du_ddd * xij
            
    # each pair was counted twice
    U *= 0.5
            
    return U, dU_dx


@nb.jit(nopython=True)
def interactions_zero_numba(x, box):
    """Lennard-Jones interactions in periodic boundary conditions.

    Evaluate potential energy and its derivatives at configuration `x` in periodic box `box`.
    This uses explicit loops which only really make sense with Numba, otherwise everything just crawls.
    """

    # prepare box inverse
    boxinv = 1.0 / box

    # allocate displacement vector
    xij = np.zeros(3)

    # accumulator for potential energy
    U = 0.0

    # accumulator for derivatives
    dU_dx = np.zeros_like(x)

    # number of atoms
    N = x.shape[0]

    # loop over all atoms but the last one
    # (last atom already has all contributions from before)
    for i in range(N-1):

        # loop over partner atoms - each pair only once
        for j in range(i+1, N):

            pass

    return U, dU_dx


@nb.jit(nopython=True)
def interactions_LJ_numba(x, box, sigma=1.0, eps=1.0):
    """Lennard-Jones interactions in periodic boundary conditions.

    Evaluate potential energy and its derivatives at configuration `x` in periodic box `box`.
    This uses explicit loops which only really make sense with Numba, otherwise everything just crawls.
    """

    # prepare box inverse
    boxinv = 1.0 / box

    # allocate displacement vector
    xij = np.zeros(3)

    # accumulator for potential energy
    U = 0.0

    # accumulator for derivatives
    dU_dx = np.zeros_like(x)

    # number of atoms
    N = x.shape[0]

    # loop over all atoms but the last one
    # (last atom already has all contributions from before)
    for i in range(N-1):

        # loop over partner atoms - each pair only once
        for j in range(i+1, N):
            #print(i,j)

            # displacement vector and distance in PBCs
            d2 = 0.0   # distance squared
            for k in range(3):
                dxk = x[j, k] - x[i, k]
                dxk -= box[k] * round(dxk * boxinv[k])    # if dxk is greater than half the box size, then subtract the box size
                xij[k] = dxk
                d2 += dxk * dxk
            #d = np.sqrt(d2)   # distance, not really needed in this version

            # this is where we would test for cutoff
            # (comparing squares is fine)
            #if d2 > r_c2:
            #    continue

            # Note that this could be generalized to any pair potential easily, perhaps in
            # a separate function, but in that case one might have to factor out that 1/r.

            # evaluate pair potential and its derivative (divided by distance)
            dinv2 = 1.0 / d2
            sdinv6 = (sigma**2 * dinv2)**3
            sdinv12 = sdinv6**2
            u = 4.0*eps * (sdinv12 - sdinv6)
            du_ddd = -24.0*eps * (2*sdinv12 - sdinv6) * dinv2

            # accumulate potential energy contribution from this pair
            U += u

            # force contribution from this pair to both atoms
            for k in range(3):
                dU_dxk = du_ddd * xij[k]
                dU_dx[i, k] -= dU_dxk
                dU_dx[j, k] += dU_dxk

    return U, dU_dx


def interactions_LJ_numpy_open(x, box, sigma=1.0, eps=1.0):
    """Lennard-Jones interactions in periodic boundary conditions.

    Evaluate potential energy and its derivatives at configuration `x` in periodic box `box`."""

    # displacement vectors
    dx = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    #print(dx)

    # distances
    d = np.sqrt((dx**2).sum(axis=2))

    # inverse distances, avoid division by zero
    dinv = np.zeros_like(d)
    idx = np.where(d != 0.0)
    dinv[idx] = 1.0 / d[idx]

    # Lennard-Jones potential energy, pairwise and total
    sdi6 = (sigma * dinv)**6
    sdi12 = sdi6**2
    U_pair = 4 * eps * (sdi12 - sdi6)
    U = 0.5 * U_pair.sum()

    # Lennard-Jones potential energy derivatives, pairwise and atomic
    #dU_dr = 24 * eps * (2 * sdi6**2 - sdi6) * dinv
    dU_dx_pair = -24 * eps * ((2 * sdi12 - sdi6) * dinv**2)[:, :, np.newaxis] * dx
    dU_dx = dU_dx_pair.sum(axis=1)

    return U, dU_dx


def interactions_LJ_numpy(x, box, sigma=1.0, eps=1.0):
    """Lennard-Jones interactions in periodic boundary conditions.

    Evaluate potential energy and its derivatives at configuration `x` in periodic box `box`."""

    # displacement vectors
    dx = x[:, np.newaxis, :] - x[np.newaxis, :, :]

    # apply PBCs to displacement vectors
    box = box[np.newaxis, np.newaxis, :]
    dx -= box * np.round(dx / box)

    # distances
    d = np.sqrt((dx**2).sum(axis=2))

    # inverse distances, avoid division by zero
    dinv = np.zeros_like(d)
    idx = np.where(d != 0.0)
    dinv[idx] = 1.0 / d[idx]

    # Lennard-Jones potential energy, pairwise and total
    sdi6 = (sigma * dinv)**6
    sdi12 = sdi6**2
    U_pair = 4 * eps * (sdi12 - sdi6)
    U = 0.5 * U_pair.sum()

    # Lennard-Jones potential energy derivatives, pairwise and atomic
    #dU_dr = 24 * eps * (2 * sdi6**2 - sdi6) * dinv
    dU_dx_pair = -24 * eps * ((2 * sdi12 - sdi6) * dinv**2)[:, :, np.newaxis] * dx
    dU_dx = dU_dx_pair.sum(axis=1)

    return U, dU_dx
