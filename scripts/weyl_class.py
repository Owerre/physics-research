# Data manipulation
import numpy as np
from scipy.sparse import diags
from numpy import pi, cos, arccos
from numpy import arctan, sin, exp, sqrt, kron
from numpy import linspace, zeros, arange 
from numpy import sort, real, meshgrid, matrix
import matplotlib.pyplot as plt

# Linear algebra
from numpy import linalg as la

class WeylMagnon:
    """
    A class for the analysis of Weyl magnon in Kagome antiferromagnets
    """
    def __init__(self, DM, Jc, h, J=1, S=1, rr = sqrt(3)):
        """ 
        Define parameters in the Hamiltonian
        """
        self.J = J
        self.S = S
        self.rr = rr
        self.DM = DM
        self.Jc = Jc
        self.h = h

    def model_hamiltonian(self, DM, Jc, h, kx, ky, kz):
        """
        The model Hamiltonian for the system

        Parameters
        _________________
        DM: DM Interaction
        Jc: Interlayer coupling
        h: External magnetic field
        kx: Momentum vector along the x direction
        ky: Momentum vector along the y direction
        kz: Momentum vector along the z direction

        Returns
        _________________
        Model Hamiltonian 
        """
        # Momentum vectors
        k1 = (-kx - self.rr * ky) / 2
        k2 = kx
        k3 = (-kx + self.rr * ky) / 2

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
        Hamiltonian = kron(sz, Gr) + kron(1j * sy, Go)
        return Hamiltonian

    def plot_weyl_bands(self, DM, Jc, h, steps):
        """
        The plot of the eigenvalues of the model Hamiltonian along the ky = 0 plane

        Parameters
        _________________
        DM: DM Interaction
        Jc: Interlayer coupling
        h: External magnetic field
        steps: Discretize steps

        Returns
        _________________
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
            # Call the Model Hamiltonian
            H = self.model_hamiltonian(DM, Jc, h, kx[ix], ky, kz)
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):

                ev1[i][ix] = eg[i]

        kz = linspace(0, pi, steps)
        for iz in arange(len(kz)):

            kx = 2 * pi / 3
            
            # Call the Model Hamiltonian
            H = self.model_hamiltonian(DM, Jc, h, kx, ky, kz[iz])
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev2[i][iz] = eg[i]

        kx1 = linspace(2 * pi / 3, 0, steps)

        # Hamiltonian diagonalization
        for ix in arange(len(kx1)):

            kz = pi
            # Call the Model Hamiltonian
            H = self.model_hamiltonian(DM, Jc, h, kx1[ix], ky, kz)
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):

                ev3[i][ix] = eg[i]

        kz1 = linspace(pi, 0, steps)
        for iz in arange(len(kz1)):

            kx = 0
            # Call the Model Hamiltonian
            H = self.model_hamiltonian(DM, Jc, h, kx, ky, kz1[iz])
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

            plt.xticks([])
            plt.text(0, -0.18, r'${\bf \Gamma}$')
            plt.text(0.24, -0.18, r'${\bf K}$')
            plt.text(0.49, -0.18, r'${\bf H}$')
            plt.text(0.74, -0.18, r'${\bf A}$')
            plt.text(0.98, -0.18, r'${\bf \Gamma}$')
            plt.axis([-0.005, 1.005, 0, 2.5])

    def weyl_surface_states_kxkz(self, DM, Jc, h, kx, kz, N):
        """ 
        This function creates Weyl magnon surface states Hamiltonain on the 010 surface.

        Parameters
        _________________
        DM: DM Interaction
        Jc: Interlayer coupling
        h: External magnetic field
        kx: Momentum vector along the x direction
        kz: Momentum vector along the z direction
        N: Number of discretize steps

        Returns
        _________________
        Tridiagonal matrix of size 6N x 6N.
        """

        hs = 6 * self.J + 2 * self.rr * self.DM + \
            4 * self.Jc  # Saturation magnetic field
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

    def plot_weyl_surface_states_kxkz(self, DM, Jc, h, steps, N):
        """ 
        This function plots the Weyl magnon surface states Hamiltonain on the 010 surface

        Parameters
        _________________
        DM: DM Interaction
        Jc: Interlayer coupling
        h: External magnetic field
        steps: Discretize steps
        N: Number of discretize steps

        Returns
        _________________
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

            # Call the Model Hamiltonian
            H = self.weyl_surface_states_kxkz(DM, Jc, h, kx[ix], kz, N)  
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(N * Hsize):
                e1[i, ix] = eg[i]

        kz1 = linspace(0, 2 * pi, steps)
        for iz in arange(len(kz1)):

            kx1 = 4 * pi / 3

            # Call the Model Hamiltonian
            H1 = self.weyl_surface_states_kxkz(DM, Jc, h, kx1, kz1[iz], N)  
            eg, ef = la.eig(H1)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(N * Hsize):
                e2[i, iz] = eg[i]

        kx2 = linspace(4 * pi / 3, 0, steps)
        for ix in arange(len(kx2)):

            kz2 = 2 * pi

            # Call the Model Hamiltonian
            H2 = self.weyl_surface_states_kxkz(DM, Jc, h, kx2[ix], kz2, N)  
            eg, ef = la.eig(H2)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(N * Hsize):
                e3[i, ix] = eg[i]

        kz3 = linspace(2 * pi, 0, steps)
        for iz in arange(len(kz3)):
            kx3 = 0

            # Call the Model Hamiltonian
            H3 = self.weyl_surface_states_kxkz( DM, Jc, h, kx3, kz3[iz], N)  
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

            plt.xticks([])
            plt.text(-0.02, -0.18, r'$(0,0)$')
            plt.text(0.2, -0.18, r'$(2\pi/3, 0)$')
            plt.text(0.4, -0.18, r'$(2\pi/3, 2\pi)$')
            plt.text(0.7, -0.18, r'$(0, 2\pi)$')
            plt.text(0.95, -0.18, r'$(0,0)$')
            plt.axis([-0.005, 1.005, 0, 2.5])