# various covariance kernels will go here
import numpy as np

# passing vector length is c style, merits vs len()?
def squared_exponential(n: int, x_i, x_j, l = 1.0):
    return sigma * np.exp((-(np.linalg.norm(x_i - x_j))**2) / (2 * l**2))

def d_squared_exponential(n: int, x_i, x_j, l = 1.0):
    term1 = -((np.linalg.norm(x_i - x_j))**2) / (2 * l**2)
    term2 = sigma * np.exp(term1)
    term3 = (x_j - x_i) / l**2
    return term1 * term2 * term3

def d_d_squared_exponential(n: int, x_i, x_j, l = 1.0):
    return

