import time
import numpy as np
from matplotlib import pyplot as plt 

def num_plotter(
        list_of_a,
        alpha = 1,
        ):
    for a in list_of_a:
        plt.hist(
            a, 
            bins = 20,
            alpha = alpha,
            edgecolor = 'k',
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

    for i in range(size):
        seed = gene_seed()
        U = pseudo_uniform(
            size = n,
            seed = seed,
            )
        B = np.multiply(U<=p, 1)
        binom = np.append(
            binom,
            [np.sum(B)]
            )
    return binom

def pseudo_poisson(
        alpha,
        size = 1,
        ):
    '''
    generate poisson
    '''
    poisson = np.array([])

    for i in range(size):
        seed = gene_seed()
        U = pseudo_uniform(
            size = 5*alpha,
            seed = seed,
            )
        Y, P, n = 0, 1, 0
        while P >= np.exp(-1*alpha):
            P = U[n]*P
            Y = Y+1
            n = n+1
        poisson = np.append(
            poisson,
            [Y],
            )
    return poisson

def pseudo_exponential(
    lmbd,
    size = 1,
    ):
    '''
    generate exponential
    '''
    seed = gene_seed()
    U = pseudo_uniform(
        seed = seed,
        size = size,
        )
    exp = -1*(1/lmbd)*(np.log(1-U))

    return exp

def box_muller(
        U1,
        U2,
        ):
    '''
    Box muller method
    '''
    a = 2*np.pi*U2
    v = np.sqrt(
        -2*np.log(U1)
        )
    return (
        v*np.cos(a),
        v*np.sin(a),
    )


def pseudo_normal(
    mu = .0,
    sigma = 1.0,
    size = 1,
    ):
    '''
    generate normal with Box-Muller transform
    '''
    seed1 = gene_seed()
    U1 = pseudo_uniform(
        seed = seed1,
        size = size,
        )

    seed2 = gene_seed()
    U2 = pseudo_uniform(
        seed = seed2,
        size = size,
        )
    
    Z1, Z2 = box_muller(
        U1, 
        U2,
        )
    
    norm = mu+Z1*sigma
    
    return norm