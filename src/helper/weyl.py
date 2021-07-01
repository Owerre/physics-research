# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import numpy as np
from scipy.sparse import diags
from numpy import pi, cos, arccos,dot,imag,gradient
from numpy import arctan, sin, exp, sqrt, kron
from numpy import linspace, zeros, arange,meshgrid
from numpy import sort, real, matrix, argsort
import matplotlib.pyplot as plt

# Linear algebra
from numpy import linalg as la

class WeylMagnon:
    """
    A class for the analysis of Weyl magnon in Kagome antiferromagnets.
    This paper is published in Phys RevB. 97, 094412, (2018).
    """
    def __init__(self, DM, Jc, h, J=1, S=1, rr = sqrt(3)):
        """ 
        Define default parameters of the model
        """
        self.J = J # Heisenberg intralayer coupling
        self.S = S # Spin value
        self.rr = rr # constant sqrt root of 3
        self.DM = DM # DM interaction
        self.Jc = Jc # Heisenberg Interlayer coupling
        self.h = h # Zeeman magnetic field strength

    def model_hamiltonian(self,k_vec):
        """
        The model Hamiltonian for the system

        Parameters
        -----------
        k_vec = (kx,ky,kz): 3D momentum vector

        Returns
        -------
        Model Hamiltonian 
        """
        # Momentum vectors
        k1 = (k_vec[0] + self.rr * k_vec[1]) / 2
        k2 = k_vec[0]
        k3 = (-k_vec[0] + self.rr * k_vec[1]) / 2
        kz = k_vec[2]

        # Saturation magnetic field and canted angle
        hs = 6 * self.J + 2 * self.rr * self.DM + 4 * self.Jc  
        th = arccos(self.h / hs)  

        # Model parameters
        t1r = self.J * (-0.5 + 3 * 0.25 * sin(th)**2) - 0.5 * \
            self.rr * self.DM * (1 - 0.5 * sin(th)**2)
        t2r = -0.5 * cos(th) * (self.J * self.rr - self.DM)
        tr = sqrt((t1r)**2 + (t2r)**2)

        to = 0.25 * (3 * self.J + self.rr * self.DM) * sin(th)**2
        trc = -self.Jc * (1 - sin(th)**2)
        toc = self.Jc * sin(th)**2

        phi = arctan(t2r / t1r)

        # Create a 6 x 6 Hamiltonian
        G0 = self.rr * self.DM + self.J + self.Jc + trc * cos(kz)

        Gr = matrix([[G0, tr * cos(k1) * exp(-1j * phi), tr * cos(k3) * exp(1j * phi)],
                     [tr * cos(k1) * exp(1j * phi), G0, tr *
                      cos(k2) * exp(-1j * phi)],
                     [tr * cos(k3) * exp(-1j * phi), tr * cos(k2) * exp(1j * phi), G0]])

        Go = matrix([[toc * cos(kz), to * cos(k1), to * cos(k3)],
                     [to * cos(k1), toc * cos(kz), to * cos(k2)],
                     [to * cos(k3), to * cos(k2), toc * cos(kz)]])

        # Pauli matrices
        sz = matrix([[1, 0], [0, -1]])
        sy = matrix([[0, -1j], [1j, 0]])

        # Model Hamiltonian
        Hk= kron(sz, Gr) + kron(1j * sy, Go)
        return Hk
    
    def velocity_xz(self, k_vec):
        """
        Velocity operator in the xz direction

        Parameters
        -----------
        k_vec = (kx,ky,kz): 3D momentum vector

        Returns
        -------
        Velocity operator
        """
        # Momentum vectors
        k1 = (k_vec[0] + self.rr * k_vec[1]) / 2
        k2 = k_vec[0]
        k3 = (-k_vec[0] + self.rr * k_vec[1]) / 2
        kz = k_vec[2]

        # Saturation magnetic field and canted angle
        hs = 6 * self.J + 2 * self.rr * self.DM + 4 * self.Jc  
        th = arccos(self.h / hs)  

        # Model parameters
        t1r = self.J * (-0.5 + 3 * 0.25 * sin(th)**2) - 0.5 * \
            self.rr * self.DM * (1 - 0.5 * sin(th)**2)
        t2r = -0.5 * cos(th) * (self.J * self.rr - self.DM)
        tr = sqrt((t1r)**2 + (t2r)**2)

        to = 0.25 * (3 * self.J + self.rr * self.DM) * sin(th)**2
        trc = -self.Jc * (1 - sin(th)**2)
        toc = self.Jc * sin(th)**2

        phi = arctan(t2r / t1r)
        
        # Pauli matrices
        sz = matrix([[1, 0], [0, -1]])
        sy = matrix([[0, -1j], [1j, 0]])

        # velocity operator along the x direction
        Grx = matrix([[0, -0.5*tr * sin(k1) * exp(-1j * phi), 0.5*tr * sin(k3) * exp(1j * phi)],
                     [-0.5*tr * sin(k1) * exp(1j * phi), 0, -tr *sin(k2) * exp(-1j * phi)],
                     [0.5*tr * sin(k3) * exp(-1j * phi), -tr * sin(k2) * exp(1j * phi), 0]]
                    )

        Gox = matrix([[0, -0.5*to * sin(k1), 0.5*to * sin(k3)],
                     [-0.5*to * sin(k1), 0, -to * sin(k2)],
                     [0.5*to * sin(k3), -to * sin(k2), 0]]
                    )

        # velocity operator along the z direction
        Grz = matrix([[ -trc * sin(kz),0,0],
                     [0,  -trc * sin(kz), 0],
                     [0, 0,  -trc * sin(kz)]]
                   )

        Goz = matrix([[-toc * sin(kz), 0, 0],
                     [0, -toc * sin(kz), 0],
                     [0, 0, -toc * sin(kz)]]
                   )
        
        # 6x6 matrices of velocity operators
        vx = kron(sz, Grx) + kron(1j * sy, Gox) 
        vz = kron(sz, Grz) + kron(1j * sy, Goz) 
        return vx, vz
     
    def berry_curvature_kxkz(self, kx, ky, kz):
        """
        Berry curvature along the xz direction

        Parameters
        -----------
        (kx,ky,kz): 3D momentum vector

        Returns
        -------
        Berry curvature of the each magnon band
        """
        for k in arange(len(kx)):
            for l in arange(len(kz)):
                k_vec = np.array([kx[k], ky, kz[l]], float)
                Hk = self.model_hamiltonian(k_vec) # Call the Hamiltonian function
                vel = self.velocity_xz(k_vec) # Call the velocity operator function
                eg, ef= la.eig(Hk) # Find the eigenvalues "eg" and eigenvectors "ef"
                idx = argsort(real(eg)) # The original index of the sorted eigenvalues
                eg = np.sort(real(eg)) # Sort the eigenvalues in ascending order
                Ene[k,l,:] = eg[:]
                eigV = ef[:,idx]

                # Compute the Berry Curvature
                for n in arange(Hsize):
                    vxvy = zeros((1,Hsize),dtype=complex)
                    for ns in arange(Hsize):
                        if ns != n:
                            a0 = dot(eigV[:,n].getH(),vel[0]) # .getH() denote complex conjugation
                            a1 = dot(eigV[:,ns].getH(),vel[1])
                            val = (dot(a0,eigV[:,ns])*dot(a1,eigV[:,n]))/(eg[n] - eg[ns])**2
                            vxvy[:,n] = vxvy[:,n] + val[0]  
                    Omega[k,l,n] = -2*imag(vxvy[:,n])
        return Omega[:,:,3], Omega[:,:,4], Omega[:,:,5]
    
    def plot_weyl_bands(self, steps):
        """
        Plot of eigenvalues of the model Hamiltonian along the ky = 0 plane

        Parameters
        ----------
        steps: Discretize steps

        Returns
        ---------
        Matplotlib line plot
        """
        Hsize = 6  # Dimension of the Hamitonian
        ky = 0  # Initialize 2D plane of the 3D Brillouin zone

        # Initial empty arrays to append the eigenvalues
        ev1 = zeros((Hsize, steps))
        ev2 = zeros((Hsize, steps))
        ev3 = zeros((Hsize, steps))
        ev4 = zeros((Hsize, steps))
        
        kx = linspace(0, 2 * pi / 3, steps)

        # Hamiltonian diagonalization
        for ix in arange(len(kx)):
            kz = 0
            k_vec = np.array([kx[ix], ky, kz], float)
            H = self.model_hamiltonian(k_vec) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order
            for i in arange(Hsize):
                ev1[i][ix] = eg[i]

        kz = linspace(0, pi, steps)
        for iz in arange(len(kz)):
            kx = 2 * pi / 3
            k_vec = np.array([kx, ky, kz[iz]], float)
            H = self.model_hamiltonian(k_vec) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev2[i][iz] = eg[i]

        kx1 = linspace(2 * pi / 3, 0, steps)

        # Hamiltonian diagonalization
        for ix in arange(len(kx1)):
            kz = pi
            k_vec = np.array([kx1[ix], ky, kz], float)
            H = self.model_hamiltonian(k_vec) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev3[i][ix] = eg[i]

        kz1 = linspace(pi, 0, steps)
        for iz in arange(len(kz1)):
            kx = 0
            k_vec = np.array([kx, ky, kz1[iz]], float)
            H = self.model_hamiltonian(k_vec) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev4[i][iz] = eg[i]

        # Plotting
        plot_x = linspace(0, 1, 4 * steps)
        for i in arange(Hsize):
            y_combined = np.concatenate((ev1[i, :], ev2[i, :], ev3[i, :], ev4[i, :]))
            plt.plot(plot_x, y_combined)
            plt.plot([0.25, 0.25], [0, 100], 'k-', linewidth=0.3)
            plt.plot([0.5, 0.5], [0, 100], 'k-', linewidth=0.3)
            plt.plot([0.75, 0.75], [0, 100], 'k-', linewidth=0.3)
            plt.text(0, -0.18, r'${\Gamma}$')
            plt.text(0.24, -0.18, r'${K}$')
            plt.text(0.49, -0.18, r'${H}$')
            plt.text(0.74, -0.18, r'${A}$')
            plt.text(0.98, -0.18, r'${\Gamma}$')
            plt.axis([-0.005, 1.005, 0, 2.5])
            plt.xticks([])

    def weyl_surface_states_kxkz(self,kx, kz, N):
        """ 
        Weyl magnon surface states Hamiltonain along the (010) surface.

        Parameters
        ----------
        kx: Momentum vector along the x direction
        kz: Momentum vector along the z direction
        N: Number of discretize steps

        Returns
        -------
        Tridiagonal matrix of size 6N x 6N.
        """

        hs = 6 * self.J + 2 * self.rr * self.DM + 4 * self.Jc  # Saturation magnetic field
        th = arccos(self.h / hs)  # Canting angle induced by magnetic field

        # Parameters
        t1r = self.J * (-0.5 + 3 * 0.25 * sin(th)**2) - 0.5 * \
            self.rr * self.DM * (1 - 0.5 * sin(th)**2)
        t2r = -0.5 * cos(th) * (self.J * self.rr - self.DM)
        tr = sqrt((t1r)**2 + (t2r)**2)

        to = 0.25 * (3 * self.J + self.rr * self.DM) * sin(th)**2
        trc = -self.Jc * (1 - sin(th)**2)
        toc = self.Jc * sin(th)**2

        phi = arctan(t2r / t1r)

        #  Matrix elements of the Hamiltonian
        G0 = self.rr * self.DM + self.J + self.Jc + trc * cos(kz)

        A11 = matrix([[G0, 0.5 * tr * exp(-1j * kx / 4) * exp(1j * phi),
         0.5 * tr * exp(1j * kx / 4) * exp(-1j * phi)],
                      [0.5 * tr * exp(1j * kx / 4) * exp(-1j * phi),
                       G0, tr * cos(kx / 2) * exp(1j * phi)],
                      [0.5 * tr * exp(-1j * kx / 4) * exp(1j * phi),
                       tr * cos(kx / 2) * exp(-1j * phi), G0]])

        A21 = matrix([[toc * cos(kz), 0.5 * to * exp(-1j * kx / 4),
         0.5 * to * exp(1j * kx / 4)],
                      [0.5 * to * exp(1j * kx / 4), toc *
                       cos(kz), to * cos(kx / 2)],
                      [0.5 * to * exp(-1j * kx / 4), to * cos(kx / 2), toc * cos(kz)]])

        # Pauli matrices
        sz = matrix([[1, 0], [0, -1]])
        sy = matrix([[0, -1j], [1j, 0]])

        # Diagonal component of the tridiagonal Hamiltonian
        H0 = kron(sz, A11) + kron(1j * sy, A21)

        B11 = matrix([[0, 0.5 * tr * exp(1j * kx / 4) * exp(1j * phi),
         0.5 * tr * exp(-1j * kx / 4) * exp(-1j * phi)],
                      [0, 0, 0], [0, 0, 0]])

        B21 = matrix([[0, 0.5 * to * exp(1j * kx / 4), 0.5 * to * exp(-1j * kx / 4)],
                      [0, 0, 0], [0, 0, 0]])

        # Off-Diagonal component of the tridiagonal Hamiltonian
        Hp = kron(sz, B11) + kron(1j * sy, B21)

        C11 = matrix([[0, 0, 0], [0.5 * tr * exp(-1j * kx / 4) * exp(-1j * phi),
                                  0, 0], [0.5 * tr * exp(1j * kx / 4) * exp(1j * phi), 0, 0]])

        C21 = matrix([[0, 0, 0], [0.5 * to * exp(-1j * kx / 4),
                                  0, 0], [0.5 * to * exp(1j * kx / 4), 0, 0]])

        # Off-Diagonal component of the tridiagonal Hamiltonian
        Hm = kron(sz, C11) + kron(1j * sy, C21)

        # Componenets of the tridiagonal matrices
        r0 = diags(np.ones(N), 0).toarray()
        rp = diags(np.ones(N - 1), 1).toarray()
        rm = diags(np.ones(N - 1), -1).toarray()

        # Tridiagonal Hamiltonian
        H_tot = np.kron(r0, H0) + np.kron(rp, Hp) + np.kron(rm, Hm)
        return H_tot

    def plot_weyl_surface_states_kxkz(self,steps, N):
        """ 
        Plot of the Weyl magnon surface states Hamiltonain along the (010) surface

        Parameters
        ----------
        steps: Discretize steps
        N: Number of discretize steps

        Returns
        -------
        Matplotlib line plot
        """
        Hsize = 6  # Dimension of the Hamitonian

        # Initial empty arrays to append the eigenvalues
        e1 = zeros((N * Hsize, steps))
        e2 = zeros((N * Hsize, steps))
        e3 = zeros((N * Hsize, steps))
        e4 = zeros((N * Hsize, steps))

        kx = linspace(0, 4 * pi / 3, steps)
        for ix in arange(len(kx)):
            kz = 0
            H = self.weyl_surface_states_kxkz(kx[ix], kz, N) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order
            for i in arange(N * Hsize):
                e1[i, ix] = eg[i]

        kz1 = linspace(0, 2 * pi, steps)
        for iz in arange(len(kz1)):
            kx1 = 4 * pi / 3
            H1 = self.weyl_surface_states_kxkz(kx1, kz1[iz], N) # Call the Model Hamiltonian  
            eg, ef = la.eig(H1)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(N * Hsize):
                e2[i, iz] = eg[i]

        kx2 = linspace(4 * pi / 3, 0, steps)
        for ix in arange(len(kx2)):
            kz2 = 2 * pi
            H2 = self.weyl_surface_states_kxkz(kx2[ix], kz2, N) # Call the Model Hamiltonian
            eg, ef = la.eig(H2)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order
            for i in arange(N * Hsize):
                e3[i, ix] = eg[i]

        kz3 = linspace(2 * pi, 0, steps)
        for iz in arange(len(kz3)):
            kx3 = 0
            H3 = self.weyl_surface_states_kxkz(kx3, kz3[iz], N)  # Call the Model Hamiltonian
            eg, ef = la.eig(H3)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(N * Hsize):
                e4[i, iz] = eg[i]

        # Plotting
        plot_x = linspace(0, 1, 4 * steps)
        for i in arange(N * Hsize):
            y_combined = np.concatenate((e1[i, :], e2[i, :], e3[i, :], e4[i, :]))
            plt.plot(plot_x, y_combined, 'b-', linewidth=0.7)
            plt.plot([0.25, 0.25], [0, 100], 'k-', linewidth=0.02)
            plt.plot([0.5, 0.5], [0, 100], 'k-', linewidth=0.02)
            plt.plot([0.75, 0.75], [0, 100], 'k-', linewidth=0.02)
            plt.text(-0.02, -0.18, r'$(0,0)$')
            plt.text(0.2, -0.18, r'$(2\pi/3, 0)$')
            plt.text(0.4, -0.18, r'$(2\pi/3, 2\pi)$')
            plt.text(0.7, -0.18, r'$(0, 2\pi)$')
            plt.text(0.95, -0.18, r'$(0,0)$')
            plt.axis([-0.005, 1.005, 0, 2.5])
            plt.xticks([])
            
    def build_U(self, vec1, vec2):
        """ 
        This function calculates the inner product of two eigenvectors divided by the norm:
        U = <psi|psi+mu>/|<psi|psi+mu>|
        
        Parameters
        ----------
        vec 1&2: vectors 1 and 2
        
        Returns
        -------
        Inner product of vec 1&2
        """
        in_product = np.dot(vec2.getH(), vec1)
        U = in_product / np.abs(in_product)
        return U

    def latF_xy(self, k_vec, Dk, dim):
        """ 
        This function calulates the lattice field in the xy direction using the definition:
        
        F12 = ln[ U1 * U2(k+1) * U1(k_2)^-1 * U2(k)^-1 ]
        
        For each k=(k1,k2,kz) point, four U must be calculated. The lattice field has the same 
        dimension of number of energy bands.
        
        Parameters
        ----------
        k_vec = (k1,k2,kz): 3D momentum vector
        Dk = (Dk1,Dk2,Dkz): 3D step vector
        dim: dimension of H(k)
        
        Returns
        -------
        lattice field corresponding to each band as an n-dimensional vector
        """
        ka = k_vec
        E, aux = la.eig(self.hamiltonian(ka))
        idx = E.real.argsort()
        E_sort = E[idx].real
        psi = aux[:, idx]

        kb = np.array([k_vec[0] + Dk[0], k_vec[1], k_vec[2]], float)
        E, aux = la.eig(self.hamiltonian(kb))
        idx = E.real.argsort()
        psiDx = aux[:, idx]

        kc = np.array([k_vec[0], k_vec[1] + Dk[1], k_vec[2]], float)
        E, aux = la.eig(self.hamiltonian(kc))
        idx = E.real.argsort()
        psiDy = aux[:, idx]

        kd = np.array([k_vec[0] + Dk[0], k_vec[1] + Dk[1], k_vec[2]], float)
        E, aux = la.eig(self.hamiltonian(kd))
        idx = E.real.argsort()
        psiDxDy = aux[:, idx]

        U1x = np.zeros((dim), dtype=complex)
        U2y = np.zeros((dim), dtype=complex)
        U1y = np.zeros((dim), dtype=complex)
        U2x = np.zeros((dim), dtype=complex)
        for i in range(dim):
            U1x[i] = self.build_U(psiDx[:, i], psi[:, i])
            U2y[i] = self.build_U(psiDy[:, i], psi[:, i])
            U1y[i] = self.build_U(psiDxDy[:, i], psiDy[:, i])
            U2x[i] = self.build_U(psiDxDy[:, i], psiDx[:, i])
        F12 = np.zeros((dim), dtype=complex)
        F12 = np.log(U1x * U2x * 1. / U1y * 1. / U2y)
        return F12
    
    def hamiltonian(self,k_vec):
        """
        The model Hamiltonian for the system

        Parameters
        -----------
        k_vec =(k1,k2,kz): 3D momentum vector
        where k1 = (kx +sqrt(3)ky)/2, k2 = kx, kz = kz
        

        Returns
        -------
        Model Hamiltonian 
        """
        # Momentum vectors
        k1 =  k_vec[1]
        k2 = k_vec[0]
        k3 = (k_vec[1] - k_vec[0])
        kz = k_vec[2]

        # Saturation magnetic field and canted angle
        hs = 6 * self.J + 2 * self.rr * self.DM + 4 * self.Jc  
        th = arccos(self.h / hs)  

        # Model parameters
        t1r = self.J * (-0.5 + 3 * 0.25 * sin(th)**2) - 0.5 * \
            self.rr * self.DM * (1 - 0.5 * sin(th)**2)
        t2r = -0.5 * cos(th) * (self.J * self.rr - self.DM)
        tr = sqrt((t1r)**2 + (t2r)**2)

        to = 0.25 * (3 * self.J + self.rr * self.DM) * sin(th)**2
        trc = -self.Jc * (1 - sin(th)**2)
        toc = self.Jc * sin(th)**2

        phi = arctan(t2r / t1r)

        # Create a 6 x 6 Hamiltonian
        G0 = self.rr * self.DM + self.J + self.Jc + trc * cos(kz)

        Gr = matrix([[G0, tr * cos(k1) * exp(-1j * phi), tr * cos(k3) * exp(1j * phi)],
                     [tr * cos(k1) * exp(1j * phi), G0, tr *
                      cos(k2) * exp(-1j * phi)],
                     [tr * cos(k3) * exp(-1j * phi), tr * cos(k2) * exp(1j * phi), G0]])

        Go = matrix([[toc * cos(kz), to * cos(k1), to * cos(k3)],
                     [to * cos(k1), toc * cos(kz), to * cos(k2)],
                     [to * cos(k3), to * cos(k2), toc * cos(kz)]])

        # Pauli matrices
        sz = matrix([[1, 0], [0, -1]])
        sy = matrix([[0, -1j], [1j, 0]])

        # Model Hamiltonian
        Hk= kron(sz, Gr) + kron(1j * sy, Go)
        return Hk