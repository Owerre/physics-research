###################################################
# Author: S. A. Owerre
# Date modified: 25/06/2021
# Class: Chern number
# Publication: Scientific Reports 9, 7197 (2019).
###################################################

# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import numpy as np
from numpy import pi, cos, arccos, arctan
from numpy import sin, exp, sqrt, kron, dot
from numpy import linspace, zeros, arange
from numpy import sort, real, meshgrid, matrix
from scipy.sparse import diags

# Linear algebra
from numpy import linalg as la

# Bessel function
from scipy.special import jv

class FloquetQSHI:
    """ 
    Class for analysing bilayer honeycomb antiferromagnets for two cases:
    Case I - Antiferromagnetically  coupled layers.
    Case II - Ferromagnetically coupled layers.
    """
    def __init__(self, Ex, Ey, omega, phi, Jc):
        """
        Define default paramaters of the model
        """
        self.Ex = Ex  # Electric field amplitude along the x direction
        self.Ey = Ey  # Electric field amplitude along the y direction
        self.omega = omega  # Angular frequency of light
        self.phi = phi  # Light polarization
        self.Jc = Jc  # Interlayer coupling
        self.J = 1  # Intralayer coupling
        self.S = 1  # Spin value
        self.rr = sqrt(3)

    def undriven_Hk1(self, k_vec):
        """
        Static undriven Hamiltonian for case I
        
        Paramters
        ---------
        k_vec: 2D momentum vector
        
        Returns
        -------
        2D Hamiltonian
        """
        # 2D momentum vectors
        k1 = self.rr * k_vec[0]
        k2 = (self.rr * k_vec[0] + 3 * k_vec[1]) / 2

        # Parameter
        vJ = self.J * self.S
        vJc = self.Jc * self.S
        v0 = 3 * vJ + vJc

        # Structure factor
        f_k = 1 + exp(1j * k1) + exp(1j * k2)
        f_ks = f_k.conj()

        # Hamiltonian
        H = matrix([[v0, 0, vJ * f_ks, vJc], [0, v0, vJc, vJ * f_k],
                    [-vJ * f_k, -vJc, -v0, 0], [-vJc, -vJ * f_ks, 0, -v0]])
        return H

    def undriven_Hk2(self, k_vec):
        """
        Static undriven Hamiltonian for case II
        
        Paramters
        ---------
        k_vec: 2D momentum vector
        
        Returns
        -------
        2D Hamiltonian
        """

        # 2D momentum vectors
        k1 = self.rr * k_vec[0]
        k2 = (self.rr * k_vec[0] + 3 * k_vec[1]) / 2

        # Parameter
        vJ = self.J * self.S
        vJc = self.Jc * self.S
        v0 = 3 * vJ + vJc

        # Structure factor
        f_k = 1 + exp(1j * k1) + exp(1j * k2)
        f_ks = f_k.conj()

        # Hamiltonian
        H = matrix([[v0, -vJ * f_ks, vJc, 0], [-vJ * f_k, v0, 0, vJc],
                    [-vJc, 0, -v0, vJ * f_k], [0, -vJc, vJ * f_ks, -v0]]
                  )
        return H
 
    def plot_undriven_magnon_band_1(self, steps):
        """
        Plot of the magnon bands for Undriven_Hamiltonian_1
        """
        Hsize = 4  # Dimension of the Hamitonian

        # Initial empty arrays to append the eigenvalues
        ev1 = zeros((Hsize, steps))
        ev2 = zeros((Hsize, steps))
        ev3 = zeros((Hsize, steps))
        
        kx = linspace(pi / self.rr, 0, steps)
        
        # Hamiltonian diagonalization
        for ix in arange(len(kx)):
            ky = -self.rr * kx[ix] / 3
            k_vec = np.array([kx[ix], ky], float)
            H = self.undriven_Hk1(k_vec)  # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev1[i, ix] = eg[i]

        kx = linspace(0, 2 * pi / (3 * self.rr), steps)
        for ix in arange(len(kx)):

            ky = self.rr * kx[ix]
            k_vec = np.array([kx[ix], ky], float)
            H = self.undriven_Hk1(k_vec)  # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev2[i, ix] = eg[i]

        kx = linspace(2 * pi / (3 * self.rr), pi / self.rr, steps)
        for ix in arange(len(kx)):
            ky = self.rr * kx[ix]
            k_vec = np.array([kx[ix], ky], float)
            H = self.undriven_Hk1(k_vec)  # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev3[i, ix] = eg[i]

        # Plotting
        plot_x = linspace(0, 1, 3 * steps)
        for i in arange(Hsize):
            y_combined = np.concatenate((ev1[i, :], ev2[i, :], ev3[i, :]))
            plt.plot(plot_x, y_combined)
            plt.plot([0.333, 0.333], [0, 100], 'k-', linewidth=0.3)
            plt.plot([0.666, 0.66], [0, 100], 'k-', linewidth=0.3)
            plt.text(-0.02, -0.2, r'${M}$')
            plt.text(0.32, -0.2, r'${\Gamma}$')
            plt.text(0.65, -0.2, r'${K}$')
            plt.text(0.98, -0.2, r'${M}$')
            plt.axis([-0.005, 1.005, 0, 4])
            plt.xticks([])

    def plot_undriven_magnon_band_2(self, steps):
        """
        Plot of the magnon bands for Undriven_Hamiltonian_2
        """

        Hsize = 4  # Dimension of the Hamitonian

        # Initial empty arrays to append the eigenvalues
        ev1 = zeros((Hsize, steps))
        ev2 = zeros((Hsize, steps))
        ev3 = zeros((Hsize, steps))

        kx = linspace(pi / self.rr, 0, steps)
        
        # Hamiltonian diagonalization
        for ix in arange(len(kx)):

            ky = -self.rr * kx[ix] / 3
            k_vec = np.array([kx[ix], ky], float)
            H = self.undriven_Hk2(k_vec)  # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev1[i, ix] = eg[i]

        kx = linspace(0, 2 * pi / (3 * self.rr), steps)
        for ix in arange(len(kx)):
            ky = self.rr * kx[ix]
            k_vec = np.array([kx[ix], ky], float)
            H = self.undriven_Hk2(k_vec)  # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev2[i, ix] = eg[i]

        kx = linspace(2 * pi / (3 * self.rr), pi / self.rr, steps)
        for ix in arange(len(kx)):
            ky = self.rr * kx[ix]
            k_vec = np.array([kx[ix], ky], float)
            H = self.undriven_Hk2(k_vec)  # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev3[i, ix] = eg[i]

        # Plotting
        plot_x = linspace(0, 1, 3 * steps)
        for i in arange(Hsize):
            y_combined = np.concatenate((ev1[i, :], ev2[i, :], ev3[i, :]))
            plt.plot(plot_x, y_combined)
            plt.plot([0.333, 0.333], [0, 100], 'k-', linewidth=0.3)
            plt.plot([0.666, 0.66], [0, 100], 'k-', linewidth=0.3)
            plt.text(-0.02, -0.3, r'${M}$')
            plt.text(0.32, -0.3, r'${\Gamma}$')
            plt.text(0.65, -0.3, r'${K}$')
            plt.text(0.98, -0.3, r'${M}$')
            plt.axis([-0.005, 1.005, 0, 7])
            plt.xticks([])

    def Floquet_Hk1(self, k_vec):
        """
        High frequency limit of the Floquet Hamiltonian for case I.
        
        Parameters
        ----------
        k_vec: 2D momentum vector
        
        Returns
        -------
        Floquet Hamiltonian in the high frequency limit
        """
        # 2D momentum vectors
        k1 = self.rr * k_vec[0]
        k2 = (self.rr * k_vec[0] + 3 * k_vec[1]) / 2

        # Parameter
        vJ = self.J * self.S
        vJc = self.Jc * self.S
        v0 = 3 * vJ + vJc

        Ep = 0.5 * sqrt(3 * self.Ey**2 + self.Ex**2 + 2 *
                        sqrt(3) * self.Ey * self.Ex * cos(self.phi))
        Em = 0.5 * sqrt(3 * self.Ey**2 + self.Ex**2 - 2 *
                        sqrt(3) * self.Ey * self.Ex * cos(self.phi))
        P1 = np.arctan(self.Ex * sin(self.phi) /
                       (self.Ey * sqrt(3) + self.Ex * cos(self.phi)))
        P2 = np.arctan(self.Ex * sin(self.phi) /
                       (self.Ey * sqrt(3) - self.Ex * cos(self.phi)))

        # Define H0
        r0 = vJ * (jv(0, self.Ex) + jv(0, Em) *exp(1j * k1) + jv(0, Ep) * exp(1j * k2))
        r0s = r0.conj()
        H0 = np.array([[v0, 0, r0s, vJc], [0, v0, vJc, r0],[-r0, -vJc, -v0, 0], [-vJc, -r0s, 0, -v0]])

        # Define H1 and H-1
        r1 = vJ * (jv(1, self.Ex) * exp(1j * self.phi) - jv(1, Em) * exp(1j * k1) * exp(-1j * P2)
                   + jv(1, Ep) * exp(1j * k2) * exp(1j * P1))

        r1s = vJ * (-jv(1, self.Ex) * exp(1j * self.phi) + jv(1, Em) * exp(-1j * k1) * exp(-1j * P2)
                    - jv(1, Ep) * exp(-1j * k2) * exp(1j * P1))

        Hp = np.array([[0, 0, r1s, 0], [0, 0, 0, r1],[-r1, 0, 0, 0], [0, -r1s, 0, 0]])
        Hm = Hp.conj().T

        # Define the Floquet Hamiltonian
        com1 = dot(H0, Hm) - dot(Hm, H0)
        com2 = dot(H0, Hp) - dot(Hp, H0)
        com3 = dot(Hm, Hp) - dot(Hp, Hm)
        Floquet_H = H0 - (1 / self.omega) * (com1 - com2 + com3)
        return Floquet_H

    def Floquet_Hk2(self, k_vec):
        """
        High frequency limit of the Floquet Hamiltonian for case II.
        
        Parameters
        ----------
        k_vec: 2D momentum vector
        
        Returns
        -------
        Floquet Hamiltonian in the high frequency limit
        """

        # 2D momentum vectors
        k1 = self.rr * k_vec[0]
        k2 = (self.rr * k_vec[0] + 3 * k_vec[1]) / 2

        # Parameter
        vJ = self.J * self.S
        vJc = self.Jc * self.S
        v0 = 3 * vJ + vJc

        Ep = 0.5 * sqrt(3 * self.Ey**2 + self.Ex**2 + 2 *
                        sqrt(3) * self.Ey * self.Ex * cos(self.phi))
        Em = 0.5 * sqrt(3 * self.Ey**2 + self.Ex**2 - 2 *
                        sqrt(3) * self.Ey * self.Ex * cos(self.phi))
        P1 = np.arctan(self.Ex * sin(self.phi) /
                       (self.Ey * sqrt(3) + self.Ex * cos(self.phi)))
        P2 = np.arctan(self.Ex * sin(self.phi) /
                       (self.Ey * sqrt(3) - self.Ex * cos(self.phi)))

        # Define H0
        r0 = vJ * (jv(0, self.Ex) + jv(0, Em) *exp(1j * k1) + jv(0, Ep) * exp(1j * k2))
        r0s = r0.conj()
        H0 = np.array([[v0, -r0s, vJc, 0], [-r0, v0, 0, vJc],
                       [-vJc, 0, -v0, r0], [0, -vJc, r0s, -v0]])

        # Define H1 and H-1
        r1 = vJ * (jv(1, self.Ex) * exp(1j * self.phi) - jv(1, Em) * exp(1j * k1) * exp(-1j * P2)
                   + jv(1, Ep) * exp(1j * k2) * exp(1j * P1))

        r1s = vJ * (- jv(1, self.Ex) * exp(1j * self.phi) + jv(1, Em) * exp(-1j * k1) * exp(-1j * P2)
                    - jv(1, Ep) * exp(-1j * k2) * exp(1j * P1))

        Hp = np.array([[0, -r1s, 0, 0], [-r1, 0, 0, 0],[0, 0, 0, r1], [0, 0, r1s, 0]])
        Hm = Hp.conj().T

        # Define the Floquet Hamiltonian
        com1 = dot(H0, Hm) - dot(Hm, H0)
        com2 = dot(H0, Hp) - dot(Hp, H0)
        com3 = dot(Hm, Hp) - dot(Hp, Hm)
        Floquet_H = H0 - (1 /self.omega) * (com1 - com2 + com3)
        return Floquet_H

    def plot_Floquet_magnon_band_1(self, steps):
        """
        Plot of the magnon bands for Floquet Hamiltonian_1.
        
        Parameters
        ----------
        steps: discretized steps
        
        Returns
        -------
        Plot of Floquet magon bands
        """
        Hsize = 4  # Dimension of the Hamitonian

        # Initial empty arrays to append the eigenvalues
        ev1 = zeros((Hsize, steps))
        ev2 = zeros((Hsize, steps))
        ev3 = zeros((Hsize, steps))

        kx = linspace(pi / self.rr, 0, steps)
        
        # Hamiltonian diagonalization
        for ix in arange(len(kx)):
            ky = -self.rr * kx[ix] / 3
            k_vec = np.array([kx[ix], ky], float)
            H = self.Floquet_Hk1(k_vec) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev1[i, ix] = eg[i]

        kx = linspace(0, 2 * pi / (3 * self.rr), steps)
        for ix in arange(len(kx)):
            ky = self.rr * kx[ix]
            k_vec = np.array([kx[ix], ky], float)
            H = self.Floquet_Hk1(k_vec) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev2[i, ix] = eg[i]

        kx = linspace(2 * pi / (3 * self.rr), pi / self.rr, steps)
        for ix in arange(len(kx)):
            ky = self.rr * kx[ix]
            k_vec = np.array([kx[ix], ky], float)
            H = self.Floquet_Hk1(k_vec)  # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev3[i, ix] = eg[i]

        # Plotting
        plot_x = linspace(0, 1, 3 * steps)
        for i in arange(Hsize):
            y_combined = np.concatenate((ev1[i, :], ev2[i, :], ev3[i, :]))
            plt.plot(plot_x, y_combined)
            plt.plot([0.333, 0.333], [0, 100], 'k-', linewidth=0.3)
            plt.plot([0.666, 0.66], [0, 100], 'k-', linewidth=0.3)
            plt.text(-0.02, -0.3, r'${M}$')
            plt.text(0.32, -0.3, r'${\Gamma}$')
            plt.text(0.65, -0.3, r'${K}$')
            plt.text(0.98, -0.3, r'${M}$')
            plt.axis([-0.005, 1.005, 0, 4])
            plt.xticks([])

    def plot_Floquet_magnon_band_2(self, steps):
        """
        Plot of the magnon bands for Floquet Hamiltonian_2
        
        Parameters
        ----------
        steps: discretized steps
        
        Returns
        -------
        Plot of Floquet magon bands
        """
        Hsize = 4  # Dimension of the Hamitonian

        # Initial empty arrays to append the eigenvalues
        ev1 = zeros((Hsize, steps))
        ev2 = zeros((Hsize, steps))
        ev3 = zeros((Hsize, steps))

        kx = linspace(pi / self.rr, 0, steps)
        
        # Hamiltonian diagonalization
        for ix in arange(len(kx)):
            ky = -self.rr * kx[ix] / 3
            k_vec = np.array([kx[ix], ky], float)
            H = self.Floquet_Hk2(k_vec) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev1[i, ix] = eg[i]

        kx = linspace(0, 2 * pi / (3 * self.rr), steps)
        for ix in arange(len(kx)):
            ky = self.rr * kx[ix]
            k_vec = np.array([kx[ix], ky], float)
            H = self.Floquet_Hk2(k_vec) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev2[i, ix] = eg[i]

        kx = linspace(2 * pi / (3 * self.rr), pi / self.rr, steps)
        for ix in arange(len(kx)):
            ky = self.rr * kx[ix]
            k_vec = np.array([kx[ix], ky], float)
            H = self.Floquet_Hk2(k_vec) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order

            for i in arange(Hsize):
                ev3[i, ix] = eg[i]

        # Plotting
        plot_x = linspace(0, 1, 3 * steps)
        for i in arange(Hsize):
            y_combined = np.concatenate((ev1[i, :], ev2[i, :], ev3[i, :]))
            plt.plot(plot_x, y_combined)
            plt.plot([0.333, 0.333], [0, 100], 'k-', linewidth=0.3)
            plt.plot([0.666, 0.66], [0, 100], 'k-', linewidth=0.3)
            plt.text(-0.02, -0.3, r'${M}$')
            plt.text(0.32, -0.3, r'${\Gamma}$')
            plt.text(0.65, -0.3, r'${K}$')
            plt.text(0.98, -0.3, r'${M}$')
            plt.axis([-0.005, 1.005, 0, 6])
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
        in_product = np.dot(vec1, vec2.conj())
        U = in_product / np.abs(in_product)
        return U

    def latF_1(self, k_vec, Dk, dim):
        """ 
        This function calulates the lattice field for Case I using the definition:
        
        F12 = ln[ U1 * U2(k+1) * U1(k_2)^-1 * U2(k)^-1 ]
        
        The lattice field has the same dimension of number of energy bands.
        
        Parameters
        ----------
        k_vec = (kx,ky): 2D momentum vector
        Dk = (Dkx,Dky): 2D step vector
        dim: dimension of H(k)
        
        Returns
        -------
        lattice field corresponding to each band as an n-dimensional vector
        """
        k = k_vec
        E, aux = la.eig(self.Floquet_Hk1(k))
        idx = E.real.argsort()
        E_sort = E[idx].real
        psi = aux[:, idx]

        k = np.array([k_vec[0] + Dk[0], k_vec[1]], float)
        E, aux = la.eig(self.Floquet_Hk1(k))
        idx = E.real.argsort()
        psiDx = aux[:, idx]

        k = np.array([k_vec[0], k_vec[1] + Dk[1]], float)
        E, aux = la.eig(self.Floquet_Hk1(k))
        idx = E.real.argsort()
        psiDy = aux[:, idx]

        k = np.array([k_vec[0] + Dk[0], k_vec[1] + Dk[1]], float)
        E, aux = la.eig(self.Floquet_Hk1(k))
        idx = E.real.argsort()
        psiDxDy = aux[:, idx]

        U1x = np.zeros((dim), dtype=complex)
        U2y = np.zeros((dim), dtype=complex)
        U1y = np.zeros((dim), dtype=complex)
        U2x = np.zeros((dim), dtype=complex)

        for i in range(dim):
            U1x[i] = self.build_U(psi[:, i], psiDx[:, i])
            U2y[i] = self.build_U(psi[:, i], psiDy[:, i])
            U1y[i] = self.build_U(psiDy[:, i], psiDxDy[:, i])
            U2x[i] = self.build_U(psiDx[:, i], psiDxDy[:, i])

        F12 = np.zeros((dim), dtype=complex)
        F12 = np.log(U1x * U2x * 1. / U1y * 1. / U2y)
        return F12, E_sort


    def latF_2(self, k_vec, Dk, dim):
        """
        This function calulates the lattice field for Case II using the definition:
        
        F12 = ln[ U1 * U2(k+1) * U1(k_2)^-1 * U2(k)^-1 ]
        
        The lattice field has the same dimension of number of energy bands.
        
        Parameters
        ----------
        k_vec = (kx,ky): 2D momentum vector
        Dk = (Dkx,Dky): 2D step vector
        dim: dimension of H(k)
        
        Returns
        -------
        lattice field corresponding to each band as an n-dimensional vector
        """
        # Here we calculate the band structure and sort them from low to high eigenenergies
        k = k_vec
        E, aux = la.eig(self.Floquet_Hk2(k))
        idx = E.real.argsort()
        E_sort = E[idx].real
        psi = aux[:, idx]

        k = np.array([k_vec[0] + Dk[0], k_vec[1]], float)
        E, aux = la.eig(self.Floquet_Hk2(k))
        idx = E.real.argsort()
        psiDx = aux[:, idx]

        k = np.array([k_vec[0], k_vec[1] + Dk[1]], float)
        E, aux = la.eig(self.Floquet_Hk2(k))
        idx = E.real.argsort()
        psiDy = aux[:, idx]

        k = np.array([k_vec[0] + Dk[0], k_vec[1] + Dk[1]], float)
        E, aux = la.eig(self.Floquet_Hk2(k))
        idx = E.real.argsort()
        psiDxDy = aux[:, idx]

        U1x = np.zeros((dim), dtype=complex)
        U2y = np.zeros((dim), dtype=complex)
        U1y = np.zeros((dim), dtype=complex)
        U2x = np.zeros((dim), dtype=complex)

        for i in range(dim):
            U1x[i] = self.build_U(psi[:, i], psiDx[:, i])
            U2y[i] = self.build_U(psi[:, i], psiDy[:, i])
            U1y[i] = self.build_U(psiDy[:, i], psiDxDy[:, i])
            U2x[i] = self.build_U(psiDx[:, i], psiDxDy[:, i])
        F12 = np.zeros((dim), dtype=complex)
        F12 = np.log(U1x * U2x * 1. / U1y * 1. / U2y)
        return F12, E_sort


    def ChernN_1(self, steps):
        """
        Chern number of the Floquet topological magnon bands for Case I
        
        Parameters
        ----------
        steps = discretized steps
        
        Returns
        -------
        Chern number of each band as an n-dimensional vector
        """
        Nd = 4
        kx_int = 0
        kx_fin = 2 * pi / (self.rr)
        Dx = (kx_fin - kx_int) / steps

        ky_int = 0
        ky_fin = 4 * pi / 3
        Dy = (ky_fin - ky_int) / steps

        Dk = np.array([Dx, Dy], float)

        LF = np.zeros((Nd), dtype=complex)
        LF_arr = np.zeros((Nd, steps, steps), dtype=float)
        E_arr = np.zeros((Nd, steps, steps), dtype=float)
        sumN = np.zeros((Nd), dtype=complex)
        E_k = np.zeros((Nd), dtype=complex)
        chernN = np.zeros((Nd), dtype=complex)

        # Loop over kx
        for ix in range(steps):
            kx = kx_int + ix * Dx

            # Loop over ky
            for iy in range(steps):
                ky = ky_int + iy * Dy
                k_vec = np.array([kx, ky], float)
                LF, E_k = self.latF_1(k_vec,Dk, Nd)
                sumN += LF

                # # save data for plotting
               # LF_arr[:,ix,iy] = LF.imag

              #  E_arr[:,ix,iy] = np.sort(E_k.real)
        chernN = sumN.imag / (2 * np.pi)
        return chernN


    def ChernN_2(self, steps):
        """
        Chern number of the Floquet topological magnon bands for Case II
        
        Parameters
        ----------
        steps = discretized steps
        
        Returns
        -------
        Chern number of each band as an n-dimensional vector
        """
        Nd = 4
        kx_int = 0
        kx_fin = 2 * pi / (self.rr)
        Dx = (kx_fin - kx_int) / steps

        ky_int = 0
        ky_fin = 4 * pi / 3
        Dy = (ky_fin - ky_int) / steps

        Dk = np.array([Dx, Dy], float)

        LF = np.zeros((Nd), dtype=complex)
        LF_arr = np.zeros((Nd, steps, steps), dtype=float)
        E_arr = np.zeros((Nd, steps, steps), dtype=float)
        sumN = np.zeros((Nd), dtype=complex)
        E_k = np.zeros((Nd), dtype=complex)
        chernN = np.zeros((Nd), dtype=complex)

        # Loop over kx
        for ix in range(steps):
            kx = kx_int + ix * Dx

            # Loop over ky
            for iy in range(steps):
                ky = ky_int + iy * Dy
                k_vec = np.array([kx, ky], float)
                LF, E_k = self.latF_2(k_vec, Dk, Nd)
                sumN += LF

                # # save data for plotting
               # LF_arr[:,ix,iy] = LF.imag

              #  E_arr[:,ix,iy] = np.sort(E_k.real)
        chernN = sumN.imag / (2 * np.pi)
        return chernN

    def Floquet_edge1(self, kx, N):
        """
        This function defines the Floquet edge states for case I.
        
        Parameters
        ----------
        N = dimension of matrix
        
        Returns
        -------
        Tridiagonal matrix
        """

        # Parameter
        vJ = self.J * self.S
        vJc = self.Jc * self.S
        v0 = 3 * vJ + vJc

        Ep = 0.5 * sqrt(3 * self.Ey**2 + self.Ex**2 + 2 *
                        sqrt(3) * self.Ey * self.Ex * cos(self.phi))
        Em = 0.5 * sqrt(3 * self.Ey**2 + self.Ex**2 - 2 *
                        sqrt(3) * self.Ey * self.Ex * cos(self.phi))
        P1 = np.arctan(self.Ex * sin(self.phi) /
                       (self.Ey * sqrt(3) + self.Ex * cos(self.phi)))
        P2 = np.arctan(self.Ex * sin(self.phi) /
                       (self.Ey * sqrt(3) - self.Ex * cos(self.phi)))

        # Define H0
        r0 = vJ * (jv(0, self.Ex) + jv(0, Em) * exp(1j * kx))

        r0s = r0.conj()
        H0 = np.array([[v0, 0, r0s, vJc], [0, v0, vJc, r0],
                       [-r0, -vJc, -v0, 0], [-vJc, -r0s, 0, -v0]])

        # Define H1 and H-1
        r1 = vJ * (jv(1, self.Ex) * exp(1j * self.phi) -
                   jv(1, Em) * exp(1j * kx) * exp(-1j * P2))

        r1s = vJ * (-jv(1, self.Ex) * exp(1j * self.phi) +
                    jv(1, Em) * exp(-1j * kx) * exp(-1j * P2))

        Hp = np.array([[0, 0, r1s, 0], [0, 0, 0, r1],
                       [-r1, 0, 0, 0], [0, -r1s, 0, 0]])

        Hm = Hp.conj().T

       # Floquet Hamiltonian for the diagonal matrix
        com1 = dot(H0, Hm) - dot(Hm, H0)
        com2 = dot(H0, Hp) - dot(Hp, H0)
        com3 = dot(Hm, Hp) - dot(Hp, Hm)
        Floquet_H0 = H0 - (1 / self.omega) * (com1 - com2 + com3)

        r10 = vJ * jv(0, Ep)
        H10 = np.array([[0, 0, 0, 0], [0, 0, 0, r10],
                        [-r10, 0, 0, 0], [0, 0, 0, 0]])
        H20 = np.array([[0, 0, r10, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, -r10, 0, 0]])

        r1p = vJ * jv(1, Ep) * exp(1j * P1)
        r1m = -vJ * jv(1, Ep) * exp(1j * P1)

        H1p = np.array([[0, 0, 0, 0], [0, 0, 0, r1p],
                        [-r1p, 0, 0, 0], [0, 0, 0, 0]])
        H1m = H1p.conj().T

        H2p = np.array([[0, 0, r1m, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, -r1m, 0, 0]])
        H2m = H2p.conj().T

        # Floquet Hamiltonian for the off-diagonal matrix
        com11 = dot(H10, H1m) - dot(H1m, H10)
        com21 = dot(H10, H1p) - dot(H1p, H10)
        com31 = dot(H1m, H1p) - dot(H1p, H1m)

        Floquet_H10 = H10 - (1 / self.omega) * (com11 - com21 + com31)

        # Floquet Hamiltonian for the off-diagonal matrix
        com12 = dot(H20, H2m) - dot(H2m, H20)
        com22 = dot(H20, H2p) - dot(H2p, H20)
        com32 = dot(H2m, H2p) - dot(H2p, H2m)

        Floquet_H20 = H20 - (1 / self.omega) * (com12 - com22 + com32)
        
        # Componenets of the tridiagonal matrices
        c0 = diags(np.ones(N), 0).toarray()
        cp = diags(np.ones(N - 1), 1).toarray()
        cm = diags(np.ones(N - 1), -1).toarray()

        # Tridiagonal Hamiltonian
        H_tot = np.kron(c0, Floquet_H0) + np.kron(cp, Floquet_H10) + np.kron(cm, Floquet_H20)
        return H_tot

    def Floquet_edge2(self, kx, N):
        """
        This function defines the Floquet edge states for case II
        
        Parameters
        ----------
        N = dimension of matrix
        
        Returns
        -------
        Tridiagonal matrix
        """

        # Parameter
        vJ = self.J * self.S
        vJc = self.Jc * self.S
        v0 = 3 * vJ + vJc

        Ep = 0.5 * sqrt(3 * self.Ey**2 + self.Ex**2 + 2 *
                        sqrt(3) * self.Ey * self.Ex * cos(self.phi))
        Em = 0.5 * sqrt(3 * self.Ey**2 + self.Ex**2 - 2 *
                        sqrt(3) * self.Ey * self.Ex * cos(self.phi))
        P1 = np.arctan(self.Ex * sin(self.phi) /
                       (self.Ey * sqrt(3) + self.Ex * cos(self.phi)))
        P2 = np.arctan(self.Ex * sin(self.phi) /
                       (self.Ey * sqrt(3) - self.Ex * cos(self.phi)))

        # Define H0
        r0 = vJ * (jv(0, self.Ex) + jv(0, Em) * exp(1j * kx))

        r0s = r0.conj()

        H0 = np.array([[v0, -r0s, vJc, 0], [-r0, v0, 0, vJc],
                       [-vJc, 0, -v0, r0], [0, -vJc, r0s, -v0]])

        # Define H1 and H-1
        r1 = vJ * (jv(1, self.Ex) * exp(1j * self.phi) -
                   jv(1, Em) * exp(1j * kx) * exp(-1j * P2))

        r1s = vJ * (-jv(1, self.Ex) * exp(1j * self.phi) +
                    jv(1, Em) * exp(-1j * kx) * exp(-1j * P2))

        Hp = np.array([[0, -r1s, 0, 0], [-r1, 0, 0, 0],
                       [0, 0, 0, r1], [0, 0, r1s, 0]])

        Hm = Hp.conj().T

        # Floquet Hamiltonian for the diagonal matrix
        com1 = dot(H0, Hm) - dot(Hm, H0)
        com2 = dot(H0, Hp) - dot(Hp, H0)
        com3 = dot(Hm, Hp) - dot(Hp, Hm)

        Floquet_H0 = H0 - (1 / self.omega) * (com1 - com2 + com3)

        r10 = vJ * jv(0, Ep)
        H10 = np.array([[0, -r10, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, r10, 0]])
        H20 = np.array([[0, 0, 0, 0], [-r10, 0, 0, 0],
                        [0, 0, 0, r10], [0, 0, 0, 0]])

        r1p = vJ * jv(1, Ep) * exp(1j * P1)
        r1m = -vJ * jv(1, Ep) * exp(1j * P1)

        H1p = np.array([[0, 0, 0, 0], [-r1p, 0, 0, 0],
                        [0, 0, r1p, 0], [0, 0, 0, 0]])
        H1m = H1p.conj().T

        H2p = np.array([[0, 0, 0, 0], [-r1m, 0, 0, 0],
                        [0, 0, 0, r1m], [0, 0, 0, 0]])
        H2m = H2p.conj().T

        # Floquet Hamiltonian for the off-diagonal matrix
        com11 = dot(H10, H1m) - dot(H1m, H10)
        com21 = dot(H10, H1p) - dot(H1p, H10)
        com31 = dot(H1m, H1p) - dot(H1p, H1m)

        Floquet_H10 = H10 - (1 / self.omega) * (com11 - com21 + com31)

        # Floquet Hamiltonian for the off-diagonal matrix
        com12 = dot(H20, H2m) - dot(H2m, H20)
        com22 = dot(H20, H2p) - dot(H2p, H20)
        com32 = dot(H2m, H2p) - dot(H2p, H2m)

        Floquet_H20 = H20 - (1 / self.omega) * (com12 - com22 + com32)

        # Componenets of the tridiagonal matrices
        c0 = diags(np.ones(N), 0).toarray()
        cp = diags(np.ones(N - 1), 1).toarray()
        cm = diags(np.ones(N - 1), -1).toarray()

        # Tridiagonal Hamiltonian
        H_tot = np.kron(c0, Floquet_H0) + np.kron(cp, Floquet_H10) + np.kron(cm, Floquet_H20)
        return H_tot

    def plot_Floquet_states_1(self, steps, N):
        """ 
        This function plots the Floquet edge states for case I
        
        Parameters
        ----------
        N = dimension of matrix
        steps: dicretied steps
        
        Returns
        -------
        Plot of the Floquet edge states for case I
        """
        Hsize = 4  # Dimension of the Hamitonian
        ev = zeros((steps, N * Hsize)) # Initial empty arrays
        kx = linspace(0, 1, steps)
        for ix in arange(len(kx)):
            H = self.Floquet_edge1(2 * pi * kx[ix], N)  # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order
            ev[ix, :] = eg
            
        # plot
        plot_x = linspace(0, 1, 4 * steps)
        plt.plot(kx, ev, 'b-', linewidth=0.7)

    def plot_Floquet_states_2(self, steps, N):
        """ 
        This function plots the Floquet edge states for case II
        
        Parameters
        ----------
        N = dimension of matrix
        steps: dicretied steps
        
        Returns
        -------
        Plot of the Floquet edge states for case II
        """
        Hsize = 4  # Dimension of the Hamitonian
        ev = zeros((steps, N * Hsize))  # Initial empty arrays
        kx = linspace(0, 1, steps)
        for ix in arange(len(kx)):
            H = self.Floquet_edge2(2 * pi * kx[ix], N) # Call the Model Hamiltonian
            eg, ef = la.eig(H)  # Find the eigenvalues and eigenvectors
            eg = sort(real(eg))  # Sort the eigenvalues in ascending order
            ev[ix, :] = eg

        # plot
        plot_x = linspace(0, 1, 4 * steps)
        plt.plot(kx, ev, 'b-', linewidth=0.7)