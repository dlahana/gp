import numpy as np
import kernels as k
import terachem_io as tcio # how to make this conditional?
import os

class GP():

    def __init__(self,  
                geom: str,
                kernel: str, 
                engine: str,
                path: str = os.getcwd(),
                l: float = 10.3, 
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
        self.E_p = 0.0
        self.X = []
        self.iteration = 0
        try:
            self.get_atoms(os.path.join(self.path, self.geom))
        except FileNotFoundError:
            print(f'Geometry file {self.geom} not found')
            print('GP exiting')
            exit()
        if self.engine == "tc":
            print("importing tcio")
            #import terachem_io as tcio
        self.current_x = np.zeros(3 * self.n)
        try:
            terachem = os.environ['TeraChem']
            print("TeraChem: " + terachem)
        except KeyError:
            print("No terachem module loaded\n")
            print('GP exiting')
            exit()
        self.U_p = np.zeros(3 * self.n + 1) # update first element (E_p) when loop starts and first energy is evaluated
    

    def get_atoms(self, geom_file: str):
        infile = open(geom_file, "r")
        print(geom_file)
        self.n = int(infile.readline())
        print("n = %d" % self.n)
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

    def build_K_xx(self, x_i, x_j):
        self.K_xx = np.zeros((3 * self.n + 1, 3 * self.n + 1))
        self.K_xx[0,0] = self.calc_k(x_i, x_j)
        self.K_xx[1:,0] = self.calc_J(x_i, x_j)  
        self.K_xx[0,1:] = np.transpose(self.K_xx[1:,0])
        self.K_xx[1:,1:] = self.calc_H(x_i, x_j)
        #if self.add_noise == True:
            #construct \Sigma_n^2 (size of K(X,X))
        return

    def build_K_xX(self, x):
        # used for evaluating the kernel distance of the current 
        # geometry from all previously encountered geometries 
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
        for i in range(self.iteration):
            self.U_p[i * (3 * n + 1)] = self.E_p
        return

    def set_new_E_p(self):
        self.E_p = np.max(self.energies)
        return

    def calc_inf_norm(self, v):
        inf_norm = 0.0
        for i in range(len(v)):
            if abs(v[i]) > inf_norm:
                inf_norm = abs(v[i])
        return inf_norm

    def calc_U_mean(self, x):
        #U = 
        return

    def calc_U_variance(self):
        return

    def minimize(self):
        tol = 1.0e-4
        # get initial energy and gradient
        tcio.launch_job(self.path)
        data = tcio.read_energy_gradient(self.n, 'out')
        self.E_p = data[0]
        self.update_U_p()
        inf_norm = self.calc_inf_norm(data[1:])
        self.Y = np.array([])
        self.energies = []
        while inf_norm > tol:
            x = tcio.read_geom(self.n, self.path, self.geom)
            if self.iteration == 0:
                self.X = np.reshape(x, (3 * self.n, 1))
            else:
                self.X = np.append(self.X, x)
            self.Y = np.append(np.array(self.Y), data)
            self.energies.append(data[0])
            self.set_new_E_p()
            self.update_U_p()
        #   find x_min for SPES using scipy lbfgs
        #   calculate E_1 and f_1 (actual quantum values at x_min)
            tcio.write_geom(self.n, self.atoms, x, self.path, self.geom)
            tcio.launch_job(self.path)  
            data1 = tcio.read_energy_gradient(self.n, 'out')
            while data1[0] > data[0]: #(I guess if SPES minimizer is greater than starting point on PES)
                self.X = np.append(self.X, x)
                self.Y = np.append(np.ndarray(self.Y), data1)
                self.energies.append(data1[0])
                self.set_new_E_p()
                self.update_U_p()
        #       find x_min for SPES using scipy lbfgs
        #       calculate E_1 and f_1 (actual quantum values at x_min)
        #       if ||f_1||_\ing > tol:
        #           break
        # x_0, E_0, f_0 <- x_1, E_1, f_1
                data[0] = 100000
            inf_norm = 0.0
        return

    def do_stuff(self):
        # bogus_x = np.zeros(3 * self.n)
        # for i in range(3 * self.n):
        #     noise = np.random.rand()
        #     if np.random.rand() > 0.5:
        #         noise = 0.0 + noise
        #     else:
        #         noise = 0.0 - noise
        #     bogus_x[i] = self.X[i,0] + noise
        #k = self.calc_k(self.X[:,0], bogus_x)
        #print(k)
        #J = self.calc_J(self.X[:,0], bogus_x)
        #print(J)
        #H = self.calc_H(self.X[:,0], bogus_x)
        #print(H)
        #self.build_K_xx(self.X[:,0], bogus_x)
        #print(self.K_xx)
        self.minimize()
        return


gp = GP("ethylene_brs.xyz", "squared_exponential", "tc", path="./gradient_examples/FOMO_CASCI/")
gp.do_stuff()