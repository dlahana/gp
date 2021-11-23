# various covariance kernels will go here
import numpy as np
from scipy.spatial.distance import cdist as cdist

# passing vector length is c style, merits vs len()?
def squared_exponential(n: int, x_i, x_j, l = 1.0, sigma = 1.0):
    #print(x_i)
    return sigma * np.exp(-((np.linalg.norm(x_i - x_j))**2) / (2 * l**2))

def d_squared_exponential(n: int, x_i, x_j, l = 1.0, sigma = 1.0):
    term1 = -((np.linalg.norm(x_i - x_j))**2) / (2 * l**2)
    term2 = sigma**2 * np.exp(term1)
    term3 = (x_i - x_j) / l**2 #confirm order of i and j
    return term2 * term3

def d_d_squared_exponential(n: int, x_i, x_j, l = 1.0, sigma = 1.0):
    # can make this faster by only doing the upper triangle, or by using real matrix operations
    H = np.zeros((3 * n, 3 * n))
    for i in range(3 * n):
        for j in range(3 * n):
            term1 = x_i[i] - x_j[i]
            term2 = (x_j[j] - x_i[j]) / (l**2)
            term3 = -((np.linalg.norm(x_i - x_j))**2) / (2 * l**2)
            term4 = np.exp(term3)
            H[i,j] = term1 * term2 * term4
            if i == j:
                H[i,j] += term4
            H[i,j] *= ((sigma**2) / (l**2))

    return H

