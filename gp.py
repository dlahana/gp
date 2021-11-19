import numpy as np
import kernels as k
import terachem_io as tcio # how to make this conditional?
import os
from scipy.optimize import minimize

class GP():

    def __init__(self,  
                geom: str,
                kernel: str, 
                engine: str,
                path: str = os.getcwd(),
                l: float = 0.6, 
                sigma_f: float = 0.1,
                sigma_n: float = 0.0002,
                add_noise: bool = True):
        
        self.kernel = kernel
        self.engine = engine
        self.path = path
        self.geom_file = os.path.join(self.path, geom)
        self.l = l
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.atoms = []
        self.add_noise = True
        self.data_points = 0
        self.E_p = 0.0
        self.X = []
        self.K_XX = np.zeros((0,0))
        self.K_X_inv = np.zeros((0,0))
        
        try:
            self.get_atoms_from_initial_geom()
        except FileNotFoundError:
            print(f'Geometry file {self.geom_file} not found')
            print('GP exiting')
            exit()
        self.current_x = np.zeros(3 * self.n)
        self.U_p = np.zeros(3 * self.n + 1) # update first element (E_p) when loop starts and first energy is evaluated
        if self.engine == "tc":
            print("importing tcio")
            #import terachem_io as tcio
        try:
            terachem = os.environ['TeraChem']
            print("TeraChem: " + terachem)
        except KeyError:
            print("No terachem module loaded\n")
            print('GP exiting')
            exit()
    
        return

    def get_atoms_from_initial_geom(self):
        infile = open(self.geom_file, "r")
        self.n = int(infile.readline())
        infile.readline()
        self.atoms = []
        for i in range(self.n):
            parsed = infile.readline().strip().split()
            self.atoms.append(parsed[0])
        infile.close()
        return 

    def read_energy_gradient(self): 
        if (self.engine == "tc"):
            data = tcio.read_energy_gradient(self.n, os.path.join(self.path,"out"), method="hf")
        return data

    def build_k_xx(self, x_i, x_j):
        k_xx = np.zeros((3 * self.n + 1, 3 * self.n + 1))
        k_xx[0,0] = self.calc_k(x_i, x_j)
        k_xx[1:,0] = self.calc_J(x_i, x_j)  
        k_xx[0,1:] = np.transpose(k_xx[1:,0])
        k_xx[1:,1:] = self.calc_H(x_i, x_j)
        #if self.add_noise == True:
            #construct \Sigma_n^2 (size of K(X,X))
        return k_xx

    def build_K_xX(self, x):
        # used for evaluating the kernel distance of the current 
        # geometry from all previously encountered geometries 
        dim = 3 * self.n + 1
        self.k_xX = np.zeros((self.data_points * dim, dim))
        for i in range(self.data_points):
            self.k_xX[dim * i: dim * i + dim, :] = self.build_k_xx(x, self.X[:, i])
        return

    def build_K_XX(self):
        if self.data_points == 1:
            self.K_XX = self.build_k_xx(self.X[:,0], self.X[:,0])
        else:
            dim = 3 * self.n + 1
            dim_full = self.data_points * (dim)
            dim_prev = (self.data_points - 1) * (dim)
            K_XX = np.zeros((dim_full, dim_full))
            K_XX[0:dim_prev, 0:dim_prev] = self.K_XX # copy over work from previous iters
            for i in range(self.data_points):
                K_XX[dim_prev:dim_full, i*dim:i*dim+dim] = self.build_k_xx(self.X[:,i], self.X[:,-1])
                K_XX[i*dim:i*dim+dim, dim_prev:dim_full] = K_XX[dim_prev:dim_full, i*dim:i*dim+dim]
            self.K_XX = K_XX
            # can I use sherman-morrison-woodsbury update to K^-1?
            # perhaps if I throw away some data to keep matrix size consistent
        return 

    def calc_K_X_inv(self):
        # add timer
        self.K_X_inv = np.linalg.inv(self.K_XX)
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
        self.U_p = np.zeros(self.data_points * (3 * self.n + 1))
        for i in range(self.data_points):
            self.U_p[i * (3 * self.n + 1)] = self.E_p
        return

    def set_new_E_p(self):
        self.E_p = np.max(self.energies) + 5
        return

    def calc_inf_norm(self, v):
        inf_norm = 0.0
        for i in range(len(v)):
            if abs(v[i]) > inf_norm:
                inf_norm = abs(v[i])
        return inf_norm

    def calc_U_mean(self, x):
        #print("bfgs x")
        #print(x)
        self.build_K_xX(x)
        U_x = self.U_p[0:3 * self.n + 1] + np.matmul(np.matmul(np.transpose(self.k_xX), self.K_X_inv), self.Y - self.U_p)
        print(U_x[1:])
        print(U_x[0])
        #return (U_x[0], U_x[1:])
        return U_x[0]

    def calc_U_variance(self):
        return

    def minimize(self):
        tol = 1.0e-4
        # get initial energy and gradient
        tcio.launch_job(self.path)
        self.data_points += 1
        data = tcio.read_energy_gradient(self.n, 'out')
        self.E_p = data[0]
        self.update_U_p()
        inf_norm = self.calc_inf_norm(data[1:])
        self.Y = np.array([])
        self.energies = []
        while inf_norm > tol:
            self.current_x = tcio.read_geom(self.n, self.geom_file)
            if self.data_points == 1:
                self.X = np.reshape(self.current_x, (3 * self.n, 1))
            else:
                self.X = np.append(self.X, np.reshape(self.current_x, (3 * self.n, 1)), axis=1)
            self.Y = np.append(np.array(self.Y), data)
            self.energies.append(data[0])
            self.set_new_E_p()
            self.update_U_p()
            self.build_K_XX() # this seems to only append onto full matrix, not rebuild, which is good
            self.calc_K_X_inv()
            #res = minimize(self.calc_U_mean, self.current_x, jac=True, method='CG') # may need to pass function arguments explicitly,
            #res = minimize(self.calc_U_mean, self.current_x, jac=True) 
            res = minimize(self.calc_U_mean, self.current_x, jac=None) 
            if res.success==False:
                print("SPES optimization failed with following status message:")
                print(res.message)
                exit()
            self.current_x = res.x 
            tcio.write_geom(self.n, self.atoms, self.current_x, self.geom_file)
            tcio.write_geom(self.n, self.atoms, self.current_x, "optim.xyz", mode="a")
            tcio.launch_job(self.path)
            self.data_points += 1  
            data1 = tcio.read_energy_gradient(self.n, 'out')
            while data1[0] > data[0]: #(I guess if SPES minimizer is greater than starting point on PES)
                self.X = np.append(self.X, self.current_x)
                self.Y = np.append(np.ndarray(self.Y), data1)
                self.energies.append(data1[0])
                self.set_new_E_p()
                self.update_U_p()
                self.build_K_XX() # this seems to only append onto full matrix, not rebuild, which is good
                self.calc_K_X_inv()
                res = minimize(self.calc_U_mean, self.current_x, jac=True, method='BFGS') # may need to pass function arguments explicitly,
                if res.success==False:
                    print("SPES optimization failed with following status message:")
                    print(res.message)
                    exit()
                self.current_x = res.x # 
                tcio.write_geom(self.n, self.atoms, self.current_x, self.geom_file)
                tcio.write_geom(self.n, self.atoms, self.current_x, "optim.xyz", mode="a")
                tcio.launch_job(self.path)
                self.data_points += 1  
        #       if ||f_1||_\inf > tol:
        #           break
                data1 = tcio.read_energy_gradient(self.n, 'out')
                inf_norm = self.calc_inf_norm(data1[1:])
                #data[0] = 100000
            data = data1
            inf_norm = self.calc_inf_norm(data[1:])
            print(f'Infinity norm = {inf_norm}')
        return

    def do_stuff(self):
     
        self.minimize()
        return


#gp = GP("ethylene_brs.xyz", "squared_exponential", "tc", path="./gradient_examples/FOMO_CASCI/")
gp = GP("h2.xyz", "squared_exponential", "tc", path="./h2/")
gp.do_stuff()