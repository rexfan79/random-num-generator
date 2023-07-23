import time
import numpy as np
from matplotlib import pyplot as plt 

def num_plotter(a):
    plt.hist(
        a, 
        bins=20, 
        edgecolor='k',
        )
    plt.show()

def gene_seed():
    '''
    generate seed from current time
    '''
    t = time.perf_counter()
    seed = int(str(t).split('.')[1])
    return seed


def pseudo_uniform(
    mult = 16807,
    mod = (2**31)-1,
    seed = 1234,
    size = 1,
    ):
    '''
    generate pseudo uniform numbers
    '''
    U = np.zeros(size)

    for n in range(size):
        if n == 0:
            x = seed
        x = (x*mult+1)%mod
        U[n] = x/mod
    
    return U

def pseudo_bernoulli(
    p = .5,
    size = 1,
    ):
    '''
    generate bernoulli
    '''
    seed = gene_seed()
    bern = pseudo_uniform(
        seed = seed,
        size = size,
        )
    bern = np.multiply(bern<=p, 1)
    return bern

def pseudo_binomial(
        n = 50,
        p = .5,
        size = 1,
        ):
    '''
    generate binomial
    '''
    binom = np.array([])

    for n in range(size):
        seed = gene_seed()
        U = pseudo_uniform(
            size = n,
            seed = seed,
            )
        B = (U<=p).astype(int)
        binom = np.append(
            binom,
            [np.sum(B)]
            )
    return binom
