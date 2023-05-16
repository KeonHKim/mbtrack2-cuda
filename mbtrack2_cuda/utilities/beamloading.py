# -*- coding: utf-8 -*-
"""
Module where the BeamLoadingEquilibrium class is defined.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.constants import c
from scipy.integrate import quad

class BeamLoadingEquilibrium():
    """Class used to compute beam equilibrium profile for a given storage ring 
    and a list of RF cavities of any harmonic. The class assumes an uniform 
    filling of the storage ring. Based on an extension of [1].

    [1] Venturini, M. (2018). Passive higher-harmonic rf cavities with general
    settings and multibunch instabilities in electron storage rings.
    Physical Review Accelerators and Beams, 21(11), 114404.

    Parameters
    ----------
    ring : Synchrotron object
    cavity_list : list of CavityResonator objects
    I0 : beam current in [A].
    auto_set_MC_theta : if True, allow class to change cavity phase for
        CavityResonator objetcs with m = 1 (i.e. main cavities)
    F : list of form factor amplitude
    PHI : list of form factor phase
    B1 : lower intergration boundary
    B2 : upper intergration boundary
    """

    def __init__(
                self, ring, cavity_list, I0, auto_set_MC_theta=False, F=None,
                PHI=None, B1=-0.2, B2=0.2):
        self.ring = ring
        self.cavity_list = cavity_list
        self.I0 = I0
        self.n_cavity = len(cavity_list)
        self.auto_set_MC_theta = auto_set_MC_theta
        if F is None:
            self.F = np.ones((self.n_cavity,))
        else:
            self.F = F
        if PHI is None:
            self.PHI = np.zeros((self.n_cavity,))
        else:
            self.PHI = PHI
        self.B1 = B1
        self.B2 = B2
        self.mpi = False
        self.__version__ = "1.0"

        # Define constants for scaled potential u(z)
        self.u0 = self.ring.U0 / (
            self.ring.ac * self.ring.sigma_delta**2
            * self.ring.E0 * self.ring.L)
        self.ug = np.zeros((self.n_cavity,))
        self.ub = np.zeros((self.n_cavity,))
        self.update_potentials()
            
    def update_potentials(self):
        """Update potentials with cavity and ring data."""
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            self.ug[i] = cavity.Vg / (
                self.ring.ac * self.ring.sigma_delta ** 2 *
                self.ring.E0 * self.ring.L * self.ring.k1 *
                cavity.m)
            self.ub[i] = 2 * self.I0 * cavity.Rs / (
                self.ring.ac * self.ring.sigma_delta**2 *
                self.ring.E0 * self.ring.L * self.ring.k1 *
                cavity.m * (1 + cavity.beta))
        
    def energy_balance(self):
        """Return energy balance for the synchronous particle
        (z = 0 ,delta = 0)."""
        delta = self.ring.U0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            delta += cavity.Vb(self.I0) * self.F[i] * np.cos(cavity.psi - self.PHI[i])
            delta -= cavity.Vg * np.cos(cavity.theta_g)
        return delta
    
    def center_of_mass(self):
        """Return center of mass position in [s]"""
        z0 = np.linspace(self.B1, self.B2, 1000)
        rho = self.rho(z0)
        CM = np.average(z0, weights=rho)
        return CM/c

    def u(self, z):
        """Scaled potential u(z)"""
        pot = self.u0 * z
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            pot += - self.ug[i] * (
                np.sin(self.ring.k1 * cavity.m * z + cavity.theta_g)
                - np.sin(cavity.theta_g))
            pot += self.ub[i] * self.F[i] * np.cos(cavity.psi) * (
                np.sin(self.ring.k1 * cavity.m * z
                       + cavity.psi - self.PHI[i])
                - np.sin(cavity.psi - self.PHI[i]))
        return pot
    
    def du_dz(self, z):
        """Partial derivative of the scaled potential u(z) by z"""
        pot = self.u0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            pot += - self.ug[i] * self.ring.k1 * cavity.m * np.cos(self.ring.k1 * cavity.m * z + cavity.theta_g)
            pot += self.ub[i] * self.F[i] * self.ring.k1 * cavity.m * np.cos(cavity.psi) * np.cos(self.ring.k1 * cavity.m * z + cavity.psi - self.PHI[i])
        return pot

    def uexp(self, z):
        return np.exp(-1 * self.u(z))

    def integrate_func(self, f, g):
        """Return Integral[f*g]/Integral[f] between B1 and B2"""
        A = quad(lambda x: f(x) * g(x), self.B1, self.B2)
        B = quad(f, self.B1, self.B2)
        return A[0] / B[0]

    def to_solve(self, x, CM=True):
        """System of non-linear equation to solve to find the form factors F
        and PHI at equilibrum.
        The system is composed of Eq. (B6) and (B7) of [1] for each cavity.
        If auto_set_MC_theta == True, the system also find the main cavity 
        phase to impose energy balance or cancel center of mass offset.
        If CM is True, the system imposes zero center of mass offset,
        if False, the system imposes energy balance.
        """
        # Update values of F, PHI and theta
        if self.auto_set_MC_theta:
            self.F = x[:-1:2]
            for i in range(self.n_cavity):
                cavity = self.cavity_list[i]
                if cavity.m == 1:
                    cavity.theta = x[-1]
                    cavity.set_generator(0.5)
                    self.update_potentials()
        else:
            self.F = x[::2]
        self.PHI = x[1::2]

        # Compute system
        if self.auto_set_MC_theta:
            res = np.zeros((self.n_cavity * 2 + 1,))
            for i in range(self.n_cavity):
                cavity = self.cavity_list[i]
                res[2 * i] = self.F[i] * np.cos(self.PHI[i]) - self.integrate_func(
                    lambda y: self.uexp(y), lambda y: np.cos(self.ring.k1 * cavity.m * y))
                res[2 * i + 1] = self.F[i] * np.sin(self.PHI[i]) - self.integrate_func(
                    lambda y: self.uexp(y), lambda y: np.sin(self.ring.k1 * cavity.m * y))
            # Factor 1e-8 or 1e12 for better convergence
            if CM is True:
                res[self.n_cavity * 2] = self.center_of_mass() * 1e12
            else:
                res[self.n_cavity * 2] = self.energy_balance() * 1e-8
        else:
            res = np.zeros((self.n_cavity * 2,))
            for i in range(self.n_cavity):
                cavity = self.cavity_list[i]
                res[2 * i] = self.F[i] * np.cos(self.PHI[i]) - self.integrate_func(
                    lambda y: self.uexp(y), lambda y: np.cos(self.ring.k1 * cavity.m * y))
                res[2 * i + 1] = self.F[i] * np.sin(self.PHI[i]) - self.integrate_func(
                    lambda y: self.uexp(y), lambda y: np.sin(self.ring.k1 * cavity.m * y))
        return res

    def rho(self, z):
        """Return bunch equilibrium profile at postion z"""
        A = quad(lambda y: self.uexp(y), self.B1, self.B2)
        return self.uexp(z) / A[0]

    def plot_rho(self, z1=None, z2=None):
        """Plot the bunch equilibrium profile between z1 and z2"""
        if z1 is None:
            z1 = self.B1
        if z2 is None:
            z2 = self.B2
        z0 = np.linspace(z1, z2, 1000)
        plt.plot(z0, self.rho(z0))
        plt.xlabel("z [m]")
        plt.title("Equilibrium bunch profile")
        
    def voltage(self, z):
        """Return the RF system total voltage at position z"""
        Vtot = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            Vtot += cavity.VRF(z, self.I0, self.F[i], self.PHI[i])
        return Vtot
    
    def dV(self, z):
        """Return derivative of the RF system total voltage at position z"""
        Vtot = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            Vtot += cavity.dVRF(z, self.I0, self.F[i], self.PHI[i])
        return Vtot
    
    def ddV(self, z):
        """Return the second derivative of the RF system total voltage at position z"""
        Vtot = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            Vtot += cavity.ddVRF(z, self.I0, self.F[i], self.PHI[i])
        return Vtot
    
    def deltaVRF(self, z):
        """Return the generator voltage minus beam loading voltage of the total RF system at position z"""
        Vtot = 0
        for i in range(self.n_cavity):
            cavity = self.cavity_list[i]
            Vtot += cavity.deltaVRF(z, self.I0, self.F[i], self.PHI[i])
        return Vtot
    
    def plot_dV(self, z1=None, z2=None):
        """Plot the derivative of RF system total voltage between z1 and z2"""
        if z1 is None:
            z1 = self.B1
        if z2 is None:
            z2 = self.B2
        z0 = np.linspace(z1, z2, 1000)
        plt.plot(z0, self.dV(z0))
        plt.xlabel("z [m]")
        plt.ylabel("Total RF voltage (V)")
        
    def plot_voltage(self, z1=None, z2=None):
        """Plot the RF system total voltage between z1 and z2"""
        if z1 is None:
            z1 = self.B1
        if z2 is None:
            z2 = self.B2
        z0 = np.linspace(z1, z2, 1000)
        plt.plot(z0, self.voltage(z0))
        plt.xlabel("z [m]")
        plt.ylabel("Total RF voltage (V)")

    def std_rho(self, z1=None, z2=None):
        """Return the rms bunch equilibrium size in [m]"""
        if z1 is None:
            z1 = self.B1
        if z2 is None:
            z2 = self.B2
        z0 = np.linspace(z1, z2, 1000)
        values = self.rho(z0)
        average = np.average(z0, weights=values)
        variance = np.average((z0 - average)**2, weights=values)
        return np.sqrt(variance)

    def beam_equilibrium(self, x0=None, tol=1e-4, method='hybr', options=None, 
                         plot = False, CM=True):
        """Solve system of non-linear equation to find the form factors F
        and PHI at equilibrum. Can be used to compute the equilibrium bunch
        profile.
        
        Parameters
        ----------
        x0 : initial guess
        tol : tolerance for termination of the algorithm
        method : method used by scipy.optimize.root to solve the system
        options : options given to scipy.optimize.root
        plot : if True, plot the equilibrium bunch profile
        CM : if True, the system imposes zero center of mass offset,
        if False, the system imposes energy balance.
        
        Returns
        -------
        sol : OptimizeResult object representing the solution
        """
        if x0 is None:
            x0 = [1, 0] * self.n_cavity
            if self.auto_set_MC_theta:
                x0 = x0 + [self.cavity_list[0].theta]

        if CM:
            print("The initial center of mass offset is " +
                  str(self.center_of_mass()*1e12) + " ps")
        else:
            print("The initial energy balance is " +
                  str(self.energy_balance()) + " eV")

        sol = root(lambda x : self.to_solve(x, CM), x0, tol=tol, method=method, 
                   options=options)

        # Update values of F, PHI and theta_g
        if self.auto_set_MC_theta:
            self.F = sol.x[:-1:2]
            for i in range(self.n_cavity):
                cavity = self.cavity_list[i]
                if cavity.m == 1:
                    cavity.theta = sol.x[-1]
        else:
            self.F = sol.x[::2]
        self.PHI = sol.x[1::2]

        if CM:
            print("The final center of mass offset is " +
                  str(self.center_of_mass()*1e12) + " ps")
        else:
            print("The final energy balance is " +
                  str(self.energy_balance()) + " eV")
        print("The algorithm has converged: " + str(sol.success))
        
        if plot:
            self.plot_rho(self.B1 / 4, self.B2 / 4)

        return sol
