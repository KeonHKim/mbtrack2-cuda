# -*- coding: utf-8 -*-
"""
This module defines the WakePotential and LongRangeResistiveWall classes which 
deal with the single bunch and multi-bunch wakes.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from scipy.constants import mu_0, c, pi
from mbtrack2.tracking.element import Element
from mbtrack2.utilities.spectrum import gaussian_bunch
   
class WakePotential(Element):
    """
    Compute a wake potential from uniformly sampled wake functions by 
    performing a convolution with a bunch charge profile.
    
    Two different time bases are used. The first one is controled by the n_bin
    parameter and is used to compute the bunch profile. Then the bunch profile
    is interpolated on the wake function time base which is used to perform the
    convolution to get the wake potential.
    
    Parameters
    ----------
    ring : Synchrotron object
    wakefield : Wakefield object
        Wakefield object which contains the wake functions to be used. The wake
        functions must be uniformly sampled!
    n_bin : int, optional
        Number of bins for constructing the longitudinal bunch profile.
        
    Attributes
    ----------
    rho : array of shape (n_bin, )
        Bunch charge density profile in the unit [1/s].
    Wp : array 
        Wake potential profile.
    Wp_interp : array of shape (mp_number, )
        Wake potential, obtained from interpolating Wp, exerted on each macro-particle.
    
    Methods
    -------
    charge_density(bunch)
        Compute bunch charge density profile in [1/s].
    dipole_moment(bunch, plane, tau0)
        Return the dipole moment of the bunch computed on the same time array 
        as the wake function.
    prepare_wakefunction(wake_type)
        Prepare the wake function of a given wake_type to be used for the wake
        potential computation. 
    get_wakepotential(bunch, wake_type)
        Return the wake potential computed on the wake function time array 
        limited to the bunch profile.
    track(bunch)
        Tracking method for the element.
    plot_last_wake(wake_type)
        Plot the last wake potential of a given type computed during the last
        call of the track method.
    reference_loss(bunch)
        Calculate the loss factor and kick factor from the wake potential and 
        compare it to a reference value assuming a Gaussian bunch computed in 
        the frequency domain.
    check_sampling()
        Check if the wake function sampling is uniform.
    reduce_sampling(factor)
        Reduce wake function samping by an integer factor.
        
    """
    
    def __init__(self, ring, wakefield, n_bin=65):
        self.wakefield = wakefield
        self.types = self.wakefield.wake_components
        self.n_types = len(self.wakefield.wake_components)
        self.ring = ring
        self.n_bin = n_bin
        self.check_sampling()
            
    def charge_density(self, bunch):
        """
        Compute bunch charge density profile in [1/s].

        Parameters
        ----------
        bunch : Bunch object

        """
        
        # Get binning data
        a, b, c, d = bunch.binning(n_bin=self.n_bin)
        self.bins = a
        self.sorted_index = b
        self.profile = c
        self.center = d
        self.bin_size = self.bins[1] - self.bins[0]
        
        # Compute charge density
        self.rho = bunch.charge_per_mp*self.profile/(self.bin_size*bunch.charge)
        self.rho = np.array(self.rho)
        
        # Compute time array
        self.tau = np.array(self.center)
        self.dtau = self.tau[1] - self.tau[0]
        
        # Add N values before and after rho and tau
        if self.n_bin % 2 == 0:
            N = int(self.n_bin/2)
            self.tau = np.arange(self.tau[0] - self.dtau*N, 
                                 self.tau[-1] + self.dtau*N,
                                 self.dtau)
            self.rho = np.append(self.rho, np.zeros(N))
            self.rho = np.insert(self.rho, 0, np.zeros(N))
        else:
            N = int(np.floor(self.n_bin/2))
            self.tau = np.arange(self.tau[0] - self.dtau*N, 
                                 self.tau[-1] + self.dtau*(N+1),
                                 self.dtau)
            self.rho = np.append(self.rho, np.zeros(N))
            self.rho = np.insert(self.rho, 0, np.zeros(N+1))
            
        if len(self.tau) != len(self.rho):
            self.tau = np.append(self.tau, self.tau[-1] + self.dtau)
            
        self.tau_mean = np.mean(self.tau)
        self.tau -= self.tau_mean
            
    def dipole_moment(self, bunch, plane, tau0):
        """
        Return the dipole moment of the bunch computed on the same time array 
        as the wake function.

        Parameters
        ----------
        bunch : Bunch object
        plane : str
            Plane on which the dipole moment is computed, "x" or "y".
        tau0 : array
            Time array on which the dipole moment will be interpolated, in [s].

        Returns
        -------
        dipole : array
            Dipole moment of the bunch.

        """
        dipole = np.zeros((self.n_bin - 1,))
        for i in range(self.n_bin - 1):
            dipole[i] = bunch[plane][self.sorted_index == i].sum()
        dipole = dipole/self.profile
        dipole[np.isnan(dipole)] = 0
        
        # Add N values to get same size as tau/profile
        if self.n_bin % 2 == 0:
            N = int(self.n_bin/2)
            dipole = np.append(dipole, np.zeros(N))
            dipole = np.insert(dipole, 0, np.zeros(N))
        else:
            N = int(np.floor(self.n_bin/2))
            dipole = np.append(dipole, np.zeros(N))
            dipole = np.insert(dipole, 0, np.zeros(N+1))
            
        # Interpole on tau0 to get the same size as W0
        dipole0 = np.interp(tau0, self.tau, dipole, 0, 0)
            
        setattr(self, "dipole_" + plane, dipole0)
        return dipole0
    
    
    def prepare_wakefunction(self, wake_type, tau, save_data=True):
        """
        Prepare the wake function of a given wake_type to be used for the wake
        potential computation. 
        
        The new time array keeps the same sampling time as given in the 
        WakeFunction definition but is restricted to the bunch profile time 
        array.

        Parameters
        ----------
        wake_type : str
            Type of the wake function to prepare: "Wlong", "Wxdip", ...
        tau : array
            Time domain array of the bunch profile in [s].
        save_data : bool, optional
            If True, the results are saved as atributes.

        Returns
        -------
        tau0 : array
            Time base of the wake function in [s].
        dtau0 : float
            Difference between two points of the wake function time base in 
            [s].
        W0 : array
            Wake function array in [V/C] or [V/C/m].

        """

        tau0 = np.array(getattr(self.wakefield, wake_type).data.index)
        dtau0 = tau0[1] - tau0[0]
        W0 = np.array(getattr(self.wakefield, wake_type).data["real"])
        
        # Keep only the wake function on the rho window
        ind = np.all([min(tau[0], 0) < tau0, max(tau[-1], 0) > tau0],
                     axis=0)
        tau0 = tau0[ind]
        W0 = W0[ind]
        
        # Check the wake function window for assymetry
        assym = (np.abs(tau0[-1]) - np.abs(tau0[0])) / dtau0
        n_assym = int(np.floor(assym))
        if np.floor(assym) > 1:
            
            # add at head
            if np.abs(tau0[-1]) >  np.abs(tau0[0]):
                tau0 = np.arange(tau0[0] - dtau0*n_assym, 
                                 tau0[-1] + dtau0, 
                                 dtau0)
                n_to_add = len(tau0) - len(W0)
                W0 = np.insert(W0, 0, np.zeros(n_to_add))
                
            # add at tail
            elif np.abs(tau0[0]) >  np.abs(tau0[-1]):
                tau0 = np.arange(tau0[0], 
                                 tau0[-1] + dtau0*(n_assym+1), 
                                 dtau0)
                n_to_add = len(tau0) - len(W0)
                W0 = np.insert(W0, 0, np.zeros(n_to_add))
                
        # Check is the wf is shorter than rho then add zeros
        if (tau0[0] > tau[0]) or (tau0[-1] < tau[-1]):
            n = max(int(np.ceil((tau0[0] - tau[0])/dtau0)),
                    int(np.ceil((tau[-1] - tau0[-1])/dtau0)))
            
            tau0 = np.arange(tau0[0] - dtau0*n, 
                             tau0[-1] + dtau0*(n+1), 
                             dtau0)
            W0 = np.insert(W0, 0, np.zeros(n))
            n_to_add = len(tau0) - len(W0)
            W0 = np.insert(W0, len(W0), np.zeros(n_to_add))

        if save_data:
            setattr(self, "tau0_" + wake_type, tau0)
            setattr(self, "dtau0_" + wake_type, dtau0)
            setattr(self, "W0_" + wake_type, W0)
            
        return (tau0, dtau0, W0)
        
    def get_wakepotential(self, bunch, wake_type):
        """
        Return the wake potential computed on the wake function time array 
        limited to the bunch profile.

        Parameters
        ----------
        bunch : Bunch object
        wake_type : str
            Wake function type: "Wlong", "Wxdip", ...

        Returns
        -------
        Wp : array
            Wake potential.

        """

        (tau0, dtau0, W0) = self.prepare_wakefunction(wake_type, self.tau)
        
        profile0 = np.interp(tau0, self.tau, self.rho, 0, 0)
        
        if wake_type == "Wlong" or wake_type == "Wxquad" or wake_type == "Wyquad":
            Wp = signal.convolve(profile0, W0*-1, mode='same')*dtau0
        elif wake_type == "Wxdip":
            dipole0 = self.dipole_moment(bunch, "x", tau0)
            Wp = signal.convolve(profile0*dipole0, W0, mode='same')*dtau0
        elif wake_type == "Wydip":
            dipole0 = self.dipole_moment(bunch, "y", tau0)
            Wp = signal.convolve(profile0*dipole0, W0, mode='same')*dtau0
        else:
            raise ValueError("This type of wake is not taken into account.")
        
        setattr(self,"profile0_" + wake_type, profile0)
        setattr(self, wake_type, Wp)
        return tau0, Wp
    
    @Element.parallel
    def track(self, bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.

        Parameters
        ----------
        bunch : Bunch or Beam object.
        
        """
        
        if len(bunch) != 0:
            self.charge_density(bunch)
            for wake_type in self.types:
                tau0, Wp = self.get_wakepotential(bunch, wake_type)
                Wp_interp = np.interp(bunch["tau"], tau0 + self.tau_mean, Wp, 0, 0)
                if wake_type == "Wlong":
                    bunch["delta"] += Wp_interp * bunch.charge / self.ring.E0
                elif wake_type == "Wxdip":
                    bunch["xp"] += Wp_interp * bunch.charge / self.ring.E0
                elif wake_type == "Wydip":
                    bunch["yp"] += Wp_interp * bunch.charge / self.ring.E0
                elif wake_type == "Wxquad":
                    bunch["xp"] += (bunch["x"] * Wp_interp * bunch.charge 
                                    / self.ring.E0)
                elif wake_type == "Wyquad":
                    bunch["yp"] += (bunch["y"] * Wp_interp * bunch.charge 
                                    / self.ring.E0)
                
    def plot_last_wake(self, wake_type, plot_rho=True, plot_dipole=False, 
                       plot_wake_function=True):
        """
        Plot the last wake potential of a given type computed during the last
        call of the track method.

        Parameters
        ----------
        wake_type : str
            Type of the wake to plot: "Wlong", "Wxdip", ...
        plot_rho : bool, optional
            Plot the normalised bunch profile. The default is True.
        plot_dipole : bool, optional
            Plot the normalised dipole moment. The default is False.
        plot_wake_function : bool, optional
            Plot the normalised wake function. The default is True.

        Returns
        -------
        fig : figure

        """
        
        labels = {"Wlong" : r"$W_{p,long}$ (V/pC)", 
                  "Wxdip" : r"$W_{p,x}^{D} (V/pC)$",
                  "Wydip" : r"$W_{p,y}^{D} (V/pC)$",
                  "Wxquad" : r"$W_{p,x}^{Q} (V/pC/m)$",
                  "Wyquad" : r"$W_{p,y}^{Q} (V/pC/m)$"}
        
        Wp = getattr(self, wake_type)
        tau0 = getattr(self, "tau0_" + wake_type)
        
        fig, ax = plt.subplots()
        ax.plot(tau0*1e12, Wp*1e-12, label=labels[wake_type])
        ax.set_xlabel("$\\tau$ (ps)")
        ax.set_ylabel(labels[wake_type])

        if plot_rho is True:
            profile0 = getattr(self,"profile0_" + wake_type)
            profile_rescaled = profile0/max(profile0)*max(np.abs(Wp))
            rho_rescaled = self.rho/max(self.rho)*max(np.abs(Wp))
            ax.plot(tau0*1e12, profile_rescaled*1e-12, label=r"$\rho$ interpolated (a.u.)")
            ax.plot((self.tau + self.tau_mean)*1e12, rho_rescaled*1e-12, label=r"$\rho$ (a.u.)", linestyle='dashed')
            plt.legend()
            
        if plot_wake_function is True:
            W0 = getattr(self, "W0_" + wake_type)
            W0_rescaled = W0/max(W0)*max(np.abs(Wp))
            ax.plot(tau0*1e12, W0_rescaled*1e-12, label=r"$W_{function}$ (a.u.)")
            plt.legend()
            
        if plot_dipole is True:
            dipole = getattr(self, "dipole_" + wake_type[1])
            dipole_rescaled = dipole/max(dipole)*max(np.abs(Wp))
            ax.plot(tau0*1e12, dipole_rescaled*1e-12, label=r"Dipole moment (a.u.)")
            plt.legend()
            
        return fig
    
    def get_gaussian_wakepotential(self, sigma, wake_type, dipole=1e-3):
        """
        Return the wake potential computed using a perfect gaussian profile.

        Parameters
        ----------
        sigma : float
            RMS bunch length in [s].
        wake_type : str
            Wake function type: "Wlong", "Wxdip", ...
        dipole : float, optional
            Dipole moment to consider in [m], (uniform dipole moment).
        Returns
        -------
        tau0 : array
            Time base in [s].
        W0 : array
            Wake function.
        Wp : array
            Wake potential.
        profile0 : array
            Gaussian bunch profile.
        dipole0 : array
            Dipole moment.

        """
        
        tau = np.linspace(-10*sigma,10*sigma, int(1e3))
        (tau0, dtau0, W0) = self.prepare_wakefunction(wake_type, tau, False)
        
        profile0 = gaussian_bunch(tau0, sigma)
        dipole0 = np.ones_like(profile0)*dipole
        
        if wake_type == "Wlong" or wake_type == "Wxquad" or wake_type == "Wyquad":
            Wp = signal.convolve(profile0, W0*-1, mode='same')*dtau0
        elif wake_type == "Wxdip":
            Wp = signal.convolve(profile0*dipole0, W0*-1, mode='same')*dtau0
        elif wake_type == "Wydip":
            Wp = signal.convolve(profile0*dipole0, W0*-1, mode='same')*dtau0
        else:
            raise ValueError("This type of wake is not taken into account.")

        return tau0, W0, Wp, profile0, dipole0
        
    def plot_gaussian_wake(self, sigma, wake_type, dipole=1e-3, plot_rho=True, 
                           plot_dipole=False, plot_wake_function=True):
        """
        Plot the wake potential of a given type for a perfect gaussian bunch.

        Parameters
        ----------
        sigma : float
            RMS bunch length in [s].
        wake_type : str
            Type of the wake to plot: "Wlong", "Wxdip", ...
        dipole : float, optional
            Dipole moment to consider in [m], (uniform dipole moment).
        plot_rho : bool, optional
            Plot the normalised bunch profile. The default is True.
        plot_dipole : bool, optional
            Plot the normalised dipole moment. The default is False.
        plot_wake_function : bool, optional
            Plot the normalised wake function. The default is True.

        Returns
        -------
        fig : figure

        """
        
        labels = {"Wlong" : r"$W_{p,long}$ (V/pC)", 
                  "Wxdip" : r"$W_{p,x}^{D} (V/pC)$",
                  "Wydip" : r"$W_{p,y}^{D} (V/pC)$",
                  "Wxquad" : r"$W_{p,x}^{Q} (V/pC/m)$",
                  "Wyquad" : r"$W_{p,y}^{Q} (V/pC/m)$"}
        
        tau0, W0, Wp, profile0, dipole0 = self.get_gaussian_wakepotential(sigma, wake_type, dipole)
        
        fig, ax = plt.subplots()
        ax.plot(tau0*1e12, Wp*1e-12, label=labels[wake_type])
        ax.set_xlabel("$\\tau$ (ps)")
        ax.set_ylabel(labels[wake_type])

        if plot_rho is True:
            profile_rescaled = profile0/max(profile0)*max(np.abs(Wp))
            ax.plot(tau0*1e12, profile_rescaled*1e-12, label=r"$\rho$ (a.u.)")
            plt.legend()
            
        if plot_wake_function is True:
            W0_rescaled = W0/max(W0)*max(np.abs(Wp))
            ax.plot(tau0*1e12, W0_rescaled*1e-12, label=r"$W_{function}$ (a.u.)")
            plt.legend()
            
        if plot_dipole is True:
            dipole_rescaled = dipole0/max(dipole0)*max(np.abs(Wp))
            ax.plot(tau0*1e12, dipole_rescaled*1e-12, label=r"Dipole moment (a.u.)")
            plt.legend()
            
        return fig
    
    def reference_loss(self, bunch):
        """
        Calculate the loss factor and kick factor from the wake potential and 
        compare it to a reference value assuming a Gaussian bunch computed in 
        the frequency domain.

        Parameters
        ----------
        bunch : Bunch object

        Returns
        -------
        loss_data : DataFrame
            An output showing the loss/kick factors compared to the reference 
            values.

        """
        loss = []
        loss_0 = []
        delta_loss = []
        index = []
        for wake_type in self.types:
            tau0, Wp = self.get_wakepotential(bunch, wake_type)
            profile0 = getattr(self, "profile0_" + wake_type)
            factorTD = np.trapz(Wp*profile0, tau0)
            
            if wake_type == "Wlong":
                factorTD *= -1
            if wake_type == "Wxdip":
                factorTD /= bunch["x"].mean()
            if wake_type == "Wydip":
                factorTD /= bunch["y"].mean()
            
            Z = getattr(self.wakefield, "Z" + wake_type[1:])
            sigma = bunch['tau'].std()
            factorFD = Z.loss_factor(sigma)
            
            loss.append(factorTD)
            loss_0.append(factorFD)
            delta_loss.append( (factorTD - factorFD) / factorFD *100 )
            if wake_type == "Wlong":
                index.append("Wlong [V/C]")
            else:
                index.append(wake_type + " [V/C/m]")
            
            column = ['TD factor', 'FD factor', 'Relative error [%]']
            
        loss_data = pd.DataFrame(np.array([loss, loss_0, delta_loss]).T, 
                                 columns=column, 
                                 index=index)
        return loss_data

    def check_sampling(self):
        """
        Check if the wake function sampling is uniform.

        Raises
        ------
        ValueError

        """
        for wake_type in self.types:
            idx = getattr(self.wakefield, wake_type).data.index
            diff = idx[1:]-idx[:-1]
            result = np.all(np.isclose(diff, diff[0], atol=1e-15))
            if result is False:
                raise ValueError("The wake function must be uniformly sampled.")
    
    def reduce_sampling(self, factor):
        """
        Reduce wake function samping by an integer factor.
        
        Used to reduce computation time for long bunches.

        Parameters
        ----------
        factor : int

        """
        for wake_type in self.types:
            idx = getattr(self.wakefield, wake_type).data.index[::factor]
            getattr(self.wakefield, wake_type).data = getattr(self.wakefield, wake_type).data.loc[idx]
        self.check_sampling()
    
    
class LongRangeResistiveWall(Element):
    """
    Element to deal with multi-bunch and multi-turn wakes from resistive wall 
    using the algorithm defined in [1].
    
    Main approximations:
        - Bunches are treated as point charge.
        - Assymptotic expression for the resistive wall wake functions are 
        used.
        - Multi-turn wakes are limited to nt turns.
    
    Self-bunch interaction is not included in this element and should be dealed
    with the WakePotential class.
    
    Parameters
    ----------
    ring : Synchrotron object
    beam : Beam object
    length : float
        Length of the resistive pipe to consider in [m].
    rho : float
        Effective resistivity to consider in [ohm.m] as in [1].
    radius : float
        Beam pipe radius to consider in [m].
    types : str or list, optional
        Wake types to consider. 
        The default is ["Wlong","Wxdip","Wydip"].
    nt : int or float, optional
        Number of turns to consider for the long range wakes. 
        The default is 50.
    x3 : float, optional
        Horizontal effective radius of the 3rd power in [m], as Eq.27 in [1].
        The default is radius.
    y3 : float, optional
        Vertical effective radius of the 3rd power in [m], as Eq.27 in [1].
        The default is radius.
    
    References
    ----------
    [1] : Skripka, Galina, et al. "Simultaneous computation of intrabunch and 
    interbunch collective beam motions in storage rings." NIM.A (2016).
    """
    def __init__(self, ring, beam, length, rho, radius, 
                 types=["Wlong","Wxdip","Wydip"], nt=50, x3=None, y3=None):
        # parameters
        self.ring = ring
        self.length = length
        self.rho = rho
        self.nt = int(nt)
        self.nb = len(beam)
        if isinstance(types, str):
            self.types = [types]
        elif isinstance(types, list):
            self.types = types
        
        # effective radius for RW
        self.radius = radius
        if x3 is not None:
            self.x3 = x3
        else:
            self.x3 = radius
        if y3 is not None:
            self.y3 = y3
        else:
            self.y3 = radius
        
        # constants
        self.Z0 = mu_0*c
        self.t0 = (2*self.rho*self.radius**2 / self.Z0)**(1/3) / c
        
        # check approximation
        if 20*self.t0 > ring.T1:
            raise ValueError("The approximated wake functions are not valid.")
        
        # init tables
        self.tau = np.ones((self.nb,self.nt))*1e100
        self.x = np.zeros((self.nb,self.nt))
        self.y = np.zeros((self.nb,self.nt))
        self.charge = np.zeros((self.nb,self.nt))
        
    def Wlong(self, t):
        """
        Approxmiate expression for the longitudinal resistive wall wake 
        function - Eq.24 of [1].

        Parameters
        ----------
        t : float
            Time in [s].

        Returns
        -------
        wl : float
            Wake function in [V/C].

        """
        wl = (1/(4*pi * self.radius) * np.sqrt(self.Z0 * self.rho / (c * pi) ) / 
              t**(3/2) ) * self.length * -1
        return wl
    
    def Wdip(self, t, plane):
        """
        Approxmiate expression for the transverse resistive wall wake 
        function - Eq.26 of [1].

        Parameters
        ----------
        t : float
            Time in [s].
        plane : str
            "x" or "y".

        Returns
        -------
        wdip : float
            Wake function in [V/C/m].

        """
        if plane == "x":
            r3 = self.x3
        elif plane == "y":
            r3 = self.y3
        else:
            raise ValueError()
            
        wdip = (1 / (pi * r3**3) * np.sqrt(self.Z0 * c * self.rho / pi) / 
                t**(1/2) * self.length)
        return wdip
    
    def update_tables(self, beam):
        """
        Update tables.
        
        Table tau[i,j] is defined as the time difference of the bunch i center 
        of mass with respect to center of the RF bucket number 0 at turn j.
        Turn 0 corresponds to the tracked turn.
        
        Positive time corresponds to past events and negative time to future 
        events.

        Parameters
        ----------
        beam : Beam object

        Returns
        -------
        None.

        """
        # shift tables to next turn
        self.tau += self.ring.T0
        self.tau = np.roll(self.tau, shift=1, axis=1)
        self.x = np.roll(self.x, shift=1, axis=1)
        self.y = np.roll(self.y, shift=1, axis=1)
        self.charge = np.roll(self.charge, shift=1, axis=1)
        
        # update tables
        if beam.mpi_switch:
            beam.mpi.share_means(beam)
            # negative sign as when bunch 0 is tracked, the others are not yet passed
            self.tau[:,0] = beam.mpi.mean_all[:,4] - beam.bunch_index*self.ring.T1
            self.x[:,0] = beam.mpi.mean_all[:,0]
            self.y[:,0] = beam.mpi.mean_all[:,2]
            self.charge[:,0] = beam.mpi.charge_all
        else:
            mean_all = beam.bunch_mean
            charge_all =  beam.bunch_charge
            # negative sign as when bunch 0 is tracked, the others are not yet passed
            self.tau[:,0] = mean_all[4, beam.filling_pattern] - beam.bunch_index*self.ring.T1
            self.x[:,0] = mean_all[0, beam.filling_pattern]
            self.y[:,0] = mean_all[2, beam.filling_pattern]
            self.charge[:,0] = charge_all[beam.filling_pattern]
        
    
    def get_kick(self, rank, wake_type):
        """
        Compute the wake kick to apply.

        Parameters
        ----------
        rank : int
            Rank of the bunch, as defined in Mpi class.
        wake_type : float
            Type of the wake to compute.

        Returns
        -------
        sum_kick : float
            Sum of the kicks from the different bunches at different turns.

        """
        sum_kick = 0
        for j in range(self.nt):
            for i in range(self.nb):
                if (j == 0) and (rank <= i):
                    continue
                deltaT = self.tau[i,j] - self.tau[rank, 0]
                if wake_type == "Wlong":
                    sum_kick += self.Wlong(deltaT) * self.charge[i,j]
                elif wake_type == "Wxdip":
                    sum_kick += self.Wdip(deltaT, "x") * self.charge[i,j] * self.x[i,j]
                elif wake_type == "Wydip":
                    sum_kick += self.Wdip(deltaT, "y") * self.charge[i,j] * self.y[i,j]
                elif wake_type == "Wxquad":
                    raise NotImplementedError()
                elif wake_type == "Wyquad":
                    raise NotImplementedError()
                    
        return sum_kick
    
    def track_bunch(self, bunch, rank):
        """
        Track a bunch.
        
        Should only be used within the track method and not standalone.

        Parameters
        ----------
        bunch : Bunch object
        rank : int
            Rank of the bunch, as defined in Mpi class.

        Returns
        -------
        None.

        """
        for wake_type in self.types:
            kick = self.get_kick(rank, wake_type)
            if wake_type == "Wlong":
                bunch["delta"] += kick / self.ring.E0
            elif wake_type == "Wxdip":
                bunch["xp"] += kick / self.ring.E0
            elif wake_type == "Wydip":
                bunch["yp"] += kick / self.ring.E0
            elif wake_type == "Wxquad":
                bunch["xp"] += (bunch["x"] * kick / self.ring.E0)
            elif wake_type == "Wyquad":
                bunch["yp"] += (bunch["y"] * kick / self.ring.E0)
    
    def track(self, beam):
        """
        Track a beam.

        Parameters
        ----------
        beam : Beam object

        Returns
        -------
        None.

        """
        self.update_tables(beam)
        
        if beam.mpi_switch:
            rank = beam.mpi.rank
            bunch_index = beam.mpi.bunch_num # Number of the tracked bunch in this processor
            bunch = beam[bunch_index]
            self.track_bunch(bunch, rank)
        else:
            for rank, bunch in enumerate(beam.not_empty):
                self.track_bunch(bunch, rank)

    