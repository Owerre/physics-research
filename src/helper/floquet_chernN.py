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

class FloquetChernN:
    """ 
    A class to compute the Chern number topological phase diagram 
    for case I - antiferromagnetically coupled layers
    """

    def __init__(self, omega, Jc):
        """
        Define default paramaters of the model
        """
        self.omega = omega  # Angular frequency of light
        self.Jc = Jc  # Interlayer coupling
        self.J = 1  # Intralayer coupling
        self.S = 1  # Spin value
        self.rr = sqrt(3)

    def Floquet_Hk_1(self, k_vec, E0_vec, phi):
        """
        High frequency limit of the Floquet Hamiltonian for Case I
        
        Parameters
        ----------
        k_vec: 2D momentum vector
        E0_vec: 2D amplitude vector
        phi: phase angle of light
        
        Returns
        -------
        Floquet Hamiltonian in the high frequency limit
        """
        # 2D momentum vectors
        k1 = k_vec[0]
        k2 = k_vec[1]

        # Parameter
        vJ = self.J * self.S
        vJc = self.Jc * self.S
        v0 = 3 * vJ + vJc

        Ex = E0_vec[0]
        Ey = E0_vec[1]

        eps = 2*10**(-16)
        Ep = 0.5 * sqrt(3 * Ey**2 + Ex**2 + 2 * sqrt(3) * Ey * Ex * cos(phi))
        Em = 0.5 * sqrt(3 * Ey**2 + Ex**2 - 2 * sqrt(3) * Ey * Ex * cos(phi))
        P1 = np.arctan(Ex * sin(phi) / ((Ey+eps) * sqrt(3) + Ex * cos(phi) + eps))
        P2 = np.arctan(Ex * sin(phi) / ((Ey+eps) * sqrt(3) - Ex * cos(phi) + eps))

        # Define H0
        r0 = vJ * (jv(0, Ex) + jv(0, Em) *
                   exp(1j * k1) + jv(0, Ep) * exp(1j * k2))
        r0s = r0.conj()
        H0 = np.array([[v0, 0, r0s, vJc], [0, v0, vJc, r0],
                       [-r0, -vJc, -v0, 0], [-vJc, -r0s, 0, -v0]])

        # Define H1 and H-1
        r1 = vJ * (jv(1, Ex) * exp(1j * phi) - jv(1, Em) * exp(1j * k1) * exp(-1j * P2)
                   + jv(1, Ep) * exp(1j * k2) * exp(1j * P1))
        r1s = vJ * (-jv(1, Ex) * exp(1j * phi) + jv(1, Em) * exp(-1j * k1) * exp(-1j * P2)
                    - jv(1, Ep) * exp(-1j * k2) * exp(1j * P1))
        Hp = np.array([[0, 0, r1s, 0], [0, 0, 0, r1],
                       [-r1, 0, 0, 0], [0, -r1s, 0, 0]])
        Hm = Hp.conj().T

      # Define the Floquet Hamiltonian
        com1 = dot(H0, Hm) - dot(Hm, H0)
        com2 = dot(H0, Hp) - dot(Hp, H0)
        com3 = dot(Hm, Hp) - dot(Hp, Hm)
        Floquet_H = H0 - (1 /self.omega) * (com1 - com2 + com3)
        return Floquet_H

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

    def latF_1(self, k_vec, E0_vec, phi, Dk, dim):
        """ 
        This function calulates the lattice field for Case I using the definition:
        
        F12 = ln[ U1 * U2(k+1) * U1(k_2)^-1 * U2(k)^-1 ]
        
        For each k=(kx,ky) point, four U must be calculated. 
        The lattice field has the same dimension of number of energy bands.
        
        Parameters
        ----------
        k_vec = (kx,ky): 2D momentum vector
        E0_vec: amplitude vector
        phi: phase angle of light
        Dk = (Dkx,Dky): 2D step vector
        dim: dimension of H(k)
        
        Returns
        -------
        lattice field corresponding to each band as an n-dimensional vector
        """
        k = k_vec
        E0 = E0_vec

        E, aux = la.eig(self.Floquet_Hk_1(k, E0, phi))
        idx = E.real.argsort()
        E_sort = E[idx].real
        psi = aux[:, idx]

        k = np.array([k_vec[0] + Dk[0], k_vec[1]], float)
        E, aux = la.eig(self.Floquet_Hk_1(k, E0, phi))
        idx = E.real.argsort()
        psiDx = aux[:, idx]

        k = np.array([k_vec[0], k_vec[1] + Dk[1]], float)
        E, aux = la.eig(self.Floquet_Hk_1(k, E0, phi))
        idx = E.real.argsort()
        psiDy = aux[:, idx]

        k = np.array([k_vec[0] + Dk[0], k_vec[1] + Dk[1]], float)
        E, aux = la.eig(self.Floquet_Hk_1(k, E0, phi))
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

    def ChernN_1(self, E0_vec, phi, steps):
        """
        Chern number of the Floquet topological magnon bands for Case I
        
        Parameters
        ----------
        E0_vec: amplitude vector
        phi: phase angle of light
        steps = numerical steps
        
        Returns
        -------
        Chern number of each band as an n-dimensional vector
        """
        Nd = 4
        kx_int = 0
        kx_fin = 2 * pi
        Dx = (kx_fin - kx_int) / steps

        ky_int = 0
        ky_fin = 2 * pi
        Dy = (ky_fin - ky_int) / steps

        Dk = np.array([Dx, Dy], float)
        LF = np.zeros((Nd), dtype=complex)
        LF_arr = np.zeros((Nd, steps, steps), dtype=float)
        E_arr = np.zeros((Nd, steps, steps), dtype=float)
        sumN = np.zeros((Nd), dtype=complex)
        E_k = np.zeros((Nd), dtype=complex)
        chernN = np.zeros((Nd), dtype=complex)

        for ix in range(steps): # Loop over kx
            kx = kx_int + ix * Dx
            for iy in range(steps): # Loop over ky
                ky = ky_int + iy * Dy
                k_vec = np.array([kx, ky], float)
                LF, E_k = self.latF_1(k_vec, E0_vec, phi, Dk, Nd)
                sumN += LF
               # LF_arr[:,ix,iy] = LF.imag # save data for plotting

              #  E_arr[:,ix,iy] = np.sort(E_k.real)
        chernN = sumN.imag / (2 * np.pi)
        return chernN

    def plot_chernN_phase(self, steps, Nk):
        """
        Plot Chern number topological phase diagram for case I 
        with circularly-polarized light and Ex = Ey

        Parameters
        ----------
        steps = numerical steps
        Nk: number of steps

        Returns
        -------
        Plot of topological phase diagram
        """
        Nd = 4  # dimension of Hamiltonain
        ChernN_arr = zeros((Nd, Nk, Nk), dtype=float)
        Ex = linspace(0, 3.4, Nk)
        phi = np.linspace(0, 2 * pi, Nk)
        for ie in range(len(Ex)): # loop over Ex = Ey
            for ip in range(len(phi)): # loop over phi
                E0_vec = np.array([Ex[ie], Ex[ie]], float)
                ChernN_arr[:, ie, ip] = self.ChernN_1(E0_vec, phi[ip], steps)
        X, Y = meshgrid(phi, Ex)
        z_min, z_max = - np.abs(ChernN_arr[0, :, :]).max(), np.abs(ChernN_arr[0, :, :]).max()
        plt.pcolormesh(X, Y, ChernN_arr[0, :, :], cmap='cool', vmin=z_min, vmax=z_max)

    def plot_chernN_phase_(self, steps, Nk):
        """
        Plot Chern number topological phase diagram for case I 
        with circularly-polarized light and Ey = sqrt(3)*Ex

        Parameters
        ----------
        steps = numerical steps
        Nk: number of steps

        Returns
        -------
        Plot of topological phase diagram
        """
        Nd = 4  # dimension of Hamiltonain
        ChernN_arr = zeros((Nd, Nk, Nk), dtype=float)
        Ex = linspace(0, 3.4, Nk)
        phi = np.linspace(0, 2 * pi, Nk)
        for ie in range(len(Ex)): # loop over Ey = sqrt(3)*Ex
            for ip in range(len(phi)): # loop over phi
                E0_vec = np.array([Ex[ie], self.rr*Ex[ie]], float)
                ChernN_arr[:, ie, ip] = self.ChernN_1(E0_vec, phi[ip], steps)
        X, Y = meshgrid(phi, Ex)
        z_min, z_max = - np.abs(ChernN_arr[0, :, :]).max(), np.abs(ChernN_arr[0, :, :]).max()
        plt.pcolormesh(X, Y, ChernN_arr[0, :, :], cmap='cool', vmin=z_min, vmax=z_max)
