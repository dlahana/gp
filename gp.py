import numpy as np
import kernels as k
import os

class GP():

    def __init__(self,  
                geom: str,
                kernel: str, 
                engine: str,
                path: str = os.getcwd(),
                l: float = 0.3, 
                sigma_f: float = 0.1,
                sigma_n: float = 0.0002,
                add_noise: bool = True):
        
        self.geom = geom
        self.kernel = kernel
        self.engine = engine
        self.path = path
        self.l = l
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.atoms = []
        self.add_noise = True
        self.data_points = 0
        try:
            self.get_starting_geom(os.path.join(self.path, self.geom))
        except FileNotFoundError:
            print(f'Geometry file {self.geom} not found')
            print('GP exiting')
            exit()
        if self.engine == "tc":
            import terachem_io as tcio
        self.current_x = np.zeros(3 * self.n)
        try:
            terachem = os.environ['TeraChem']
            print("TeraChem: " + terachem)
        except KeyError:
            print("No terachem module loaded\n")
            print('GP exiting')
            exit()
        self.E_p = 1.0 # replace this with value from first energy evaluation
        self.U_p = np.zeros(3 * self.n + 1)
        self.U_p[0] = self.E_p
        #print(self.X)
    

    def get_starting_geom(self, geom_file: str):
        infile = open(geom_file, "r")
        self.n = int(infile.readline())
        x = np.zeros(3 * self.n)
        infile.readline()
        for i in range(self.n):
            parsed = infile.readline().strip().split()
            self.atoms.append(parsed[0])
            x[3 * i + 0] = parsed[1]
            x[3 * i + 1] = parsed[2]
            x[3 * i + 2] = parsed[3]
        self.X = np.array(x)
        self.X = np.reshape(self.X, (3 * self.n, 1))

        return 


    def read_energy_gradient(self): 
        if (self.engine == "tc"):
            data = tcio.read_energy_gradient(self.n, os.path.join(self.path,"out"), method="hf")
        
        return data


    def build_K_xx(self, x_i, x_j):
        self.K_xx = np.zeros((3 * self.n + 1, 3 * self.n + 1))
        self.K_xx[0,0] = self.calc_k(x_i, x_j)
        self.K_xx[1:,0] = self.calc_J(x_i, x_j)  
        self.K_xx[0,:1] = np.transpose(self.K_xx[1:,0])
        self.K_xx[1:,1:] = self.calc_H(x_i, x_j)
        #if self.add_noise == True:
            #construct \Sigma_n^2 (size of K(X,X))
        
        return

    def build_K_xX(self, x):
        self.k_xX = np.zeros((self.data_points * (3 * self.n + 1), 3 * self.n + 1))
        for i in range(self.data_points):
            self.k_xX[(3 * self.n + 1) * i: (3 * self.n + 1) * (i + 1), :] = self.build_K_xx(x, self.X[i, :]) 
        
        return

    def build_K_XX(self):
        # dimension (p x p)
        # p = data_points * (3 * n + 1)
        self.K_XX = np.append(self.K_XX, np.zeros((self.data_points * (3 * self.n + 1)), 0))
        self.K_XX = np.append(self.K_XX, np.zeros(((self.data_points + 1) * (3 * self.n + 1)), 1))
        self.K_XX[self.data_points * (3 * self.n + 1):, 0:self.data_points * (3 * self.n +1)] = self.k_xX
        self.K_XX[0:self.data_points * (3 * self.n + 1), self.data_points * (3 * self.n +1):] = np.transpose(self.k_xX)
        self.K_XX[self.data_points * (3 * self.n + 1):, self.data_points * (3 * self.n + 1):] = self.k_xx(self.current_x, self.current_x)
        # can I use sherman-morrison-woodsbury update to K^-1?
        # perhaps if I throw away some data to keep matrix size consistent
        return 

    def invert_K_X(self):
        self.K_inv = np.linalg.inv(self.K_XX)
        return


    def calc_k(self, x_i, x_j):
        # wrapper for covariance function
        # do this with function pointers or similar later?
        # set k, J, H functions in __init__?
        if (self.kernel == "squared_exponential"):
            return k.squared_exponential(self.n, x_i, x_j, self.l, self.sigma_f)
        else:
            return


    def calc_J(self, x_i, x_j):
        # wrapper for covariance function first derivative
        if (self.kernel == "squared_exponential"):
            return k.d_squared_exponential(self.n, x_i, x_j, self.l, self.sigma_f)
        else:
            return
    
    def calc_H(self, x_i, x_j):
        # wrapper for covariance function second derivative
        if (self.kernel == "squared_exponential"):
            return k.d_d_squared_exponential(self.n, x_i, x_j, self.l, self.sigma_f)
        else:
            return

    def update_U_p(self):
        self.U_p[0] = self.E_p
        return

    def calc_U_mean(self):
        return

    def calc_U_variance(self):
        return

    def update_prior(self):
        return

    def minimize(self):
        return

    def do_stuff(self):
        k = self.calc_k(self.X[0,:], self.X[0,:] - np.ones(3 * self.n))
        print(k)
        return


gp = GP("ethylene_brs4.xyz", "squared_exponential", "tc", path="./gradient_examples/FOMO_CASCI/")
gp.do_stuff()