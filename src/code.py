import numpy as np
from matplotlib import pyplot as plt 

def num_plotter(a):
    plt.hist(
        a, 
        bins=20, 
        edgecolor='k',
        )
    plt.show()


def gene_uniform(
    mult = 16807,
    mod = (2**31)-1,
    seed = 1234,
    size = 1,
    ):
    '''
    generate pseudo uniform numbers
    '''
    U = np.zeros(size)

    for n in range(0, size):
        if n == 0:
            x = seed
        x = (x*mult+1)%mod
        U[n] = x/mod
    
    return U