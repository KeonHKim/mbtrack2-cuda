# -*- coding: utf-8 -*-
"""
Module where the Synchrotron class is defined.
"""

import numpy as np
from scipy.constants import c, e
        
class Synchrotron:
    """
    Synchrotron class to store main properties.
    
    Optional parameters are optional only if the Optics object passed to the 
    class uses a loaded lattice.
    
    Parameters
    ----------
    h : int
        Harmonic number of the accelerator.
    optics : Optics object
        Object where the optic functions are stored.
    particle : Particle object
        Particle which is accelerated.
    tau : array of shape (3,)
        Horizontal, vertical and longitudinal damping times in [s].
    sigma_delta : float
        Equilibrium energy spread.
    sigma_0 : float
        Natural bunch length in [s].
    emit : array of shape (2,)
        Horizontal and vertical equilibrium emittance in [m.rad].
    L : float, optional
        Ring circumference in [m].
    E0 : float, optional
        Nominal (total) energy of the ring in [eV].
    ac : float, optional
        Momentum compaction factor.
    tune : array of shape (2,), optional
        Horizontal and vertical tunes.
    chro : array of shape (2,), optional
        Horizontal and vertical (non-normalized) chromaticities.
    U0 : float, optional
        Energy loss per turn in [eV].
    adts : list of arrays or None, optional
        List that contains arrays of polynomial's coefficients, in decreasing 
        powers, used to determine Amplitude-Dependent Tune Shifts (ADTS). 
        The order of the elements strictly needs to be
        [coef_xx, coef_yx, coef_xy, coef_yy], where x and y denote the horizontal
        and the vertical plane, respectively, and coef_PQ means the polynomial's
        coefficients of the ADTS in plane P due to the offset in plane Q.

        For example, if the tune shift in y due to the offset x is characterized
        by the equation dQy(x) = 3*x**2 + 2*x + 1, then coef_yx takes the form
        np.array([3, 2, 1]).
        
        Use None, to exclude the ADTS calculation.
        
    Attributes
    ----------
    T0 : float
        Revolution time in [s].
    f0 : float
        Revolution frequency in [Hz].
    omega0 : float
        Angular revolution frequency in [Hz.rad]
    T1 : flaot
        Fundamental RF period in [s].
    f1 : float
        Fundamental RF frequency in [Hz].
    omega1 : float
        Fundamental RF angular frequency in [Hz.rad].
    k1 : float
        Fundamental RF wave number in [m**-1].
    gamma : float
        Relativistic Lorentz gamma.
    beta : float
        Relativistic Lorentz beta.
    eta : float
        Momentum compaction.
    sigma : array of shape (4,)
        RMS beam size at equilibrium in [m].
        
    Methods
    -------
    synchrotron_tune(Vrf)
        Compute synchrotron tune from RF voltage.
    sigma(position)
    """
    def __init__(self, h, optics, particle, **kwargs):
        self._h = h
        self.particle = particle
        self.optics = optics
        
        if self.optics.use_local_values == False:
            self.L = kwargs.get('L', self.optics.lattice.circumference)
            self.E0 = kwargs.get('E0', self.optics.lattice.energy)
            self.ac = kwargs.get('ac', self.optics.ac)
            self.tune = kwargs.get('tune', self.optics.tune)
            self.chro = kwargs.get('chro', self.optics.chro)
            self.U0 = kwargs.get('U0', self.optics.lattice.energy_loss)
        else:
            self.L = kwargs.get('L') # Ring circumference [m]
            self.E0 = kwargs.get('E0') # Nominal (total) energy of the ring [eV]
            self.ac = kwargs.get('ac') # Momentum compaction factor
            self.tune = kwargs.get('tune') # X/Y/S tunes
            self.chro = kwargs.get('chro') # X/Y (non-normalized) chromaticities
            self.U0 = kwargs.get('U0') # Energy loss per turn [eV]
            
        self.tau = kwargs.get('tau') # X/Y/S damping times [s]
        self.sigma_delta = kwargs.get('sigma_delta') # Equilibrium energy spread
        self.sigma_0 = kwargs.get('sigma_0') # Natural bunch length [s]
        self.emit = kwargs.get('emit') # X/Y emittances in [m.rad]
        self.adts = kwargs.get('adts') # Amplitude-Dependent Tune Shift (ADTS)
                
    @property
    def h(self):
        """Harmonic number"""
        return self._h
    
    @h.setter
    def h(self, value):
        self._h = value
        self.L = self.L  # call setter
        
    @property
    def L(self):
        """Ring circumference [m]"""
        return self._L
    
    @L.setter
    def L(self,value):
        self._L = value
        self._T0 = self.L/c
        self._T1 = self.T0/self.h
        self._f0 = 1/self.T0
        self._omega0 = 2*np.pi*self.f0
        self._f1 = self.h*self.f0
        self._omega1 = 2*np.pi*self.f1
        self._k1 = self.omega1/c
        
    @property
    def T0(self):
        """Revolution time [s]"""
        return self._T0
    
    @T0.setter
    def T0(self, value):
        self.L = c*value
        
    @property
    def T1(self):
        """"Fundamental RF period [s]"""
        return self._T1
    
    @T1.setter
    def T1(self, value):
        self.L = c*value*self.h
        
    @property
    def f0(self):
        """Revolution frequency [Hz]"""
        return self._f0
    
    @f0.setter
    def f0(self,value):
        self.L = c/value
        
    @property
    def omega0(self):
        """Angular revolution frequency [Hz rad]"""
        return self._omega0
    
    @omega0.setter
    def omega0(self,value):
        self.L = 2*np.pi*c/value
        
    @property
    def f1(self):
        """Fundamental RF frequency [Hz]"""
        return self._f1
    
    @f1.setter
    def f1(self,value):
        self.L = self.h*c/value
        
    @property
    def omega1(self):
        """Fundamental RF angular frequency[Hz rad]"""
        return self._omega1
    
    @omega1.setter
    def omega1(self,value):
        self.L = 2*np.pi*self.h*c/value
        
    @property
    def k1(self):
        """Fundamental RF wave number [m**-1]"""
        return self._k1
    
    @k1.setter
    def k1(self,value):
        self.L = 2*np.pi*self.h/value
    
    @property
    def gamma(self):
        """Relativistic gamma"""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self._beta = np.sqrt(1 - self.gamma**-2)
        self._E0 = self.gamma*self.particle.mass*c**2/e

    @property
    def beta(self):
        """Relativistic beta"""
        return self._beta

    @beta.setter
    def beta(self, value):
        self.gamma = 1/np.sqrt(1-value**2)
        
    @property
    def E0(self):
        """Nominal (total) energy of the ring [eV]"""
        return self._E0
    
    @E0.setter
    def E0(self, value):
        self.gamma = value/(self.particle.mass*c**2/e)

    @property
    def eta(self):
        """Momentum compaction"""
        return self.ac - 1/(self.gamma**2)
    
    def sigma(self, position=None):
        """
        Return the RMS beam size at equilibrium in [m].

        Parameters
        ----------
        position : float or array, optional
            Longitudinal position in [m] where the beam size is computed. 
            If None, the local values are used.

        Returns
        -------
        sigma : array
            RMS beam size in [m] at position location or at local positon if 
            position is None.

        """
        if position is None:
            sigma = np.zeros((4,))
            sigma[0] = (self.emit[0]*self.optics.local_beta[0] +
                        self.optics.local_dispersion[0]**2*self.sigma_delta**2)**0.5
            sigma[1] = (self.emit[0]*self.optics.local_gamma[0] +
                        self.optics.local_dispersion[1]**2*self.sigma_delta**2)**0.5
            sigma[2] = (self.emit[1]*self.optics.local_beta[1] +
                        self.optics.local_dispersion[2]**2*self.sigma_delta**2)**0.5
            sigma[3] = (self.emit[1]*self.optics.local_gamma[1] +
                        self.optics.local_dispersion[3]**2*self.sigma_delta**2)**0.5
        else:
            if isinstance(position, (float, int)):
                n = 1
            else:
                n = len(position)
            sigma = np.zeros((4, n))
            sigma[0,:] = (self.emit[0]*self.optics.beta(position)[0] +
                        self.optics.dispersion(position)[0]**2*self.sigma_delta**2)**0.5
            sigma[1,:] = (self.emit[0]*self.optics.gamma(position)[0] +
                        self.optics.dispersion(position)[1]**2*self.sigma_delta**2)**0.5
            sigma[2,:] = (self.emit[1]*self.optics.beta(position)[1] +
                        self.optics.dispersion(position)[2]**2*self.sigma_delta**2)**0.5
            sigma[3,:] = (self.emit[1]*self.optics.gamma(position)[1] +
                        self.optics.dispersion(position)[3]**2*self.sigma_delta**2)**0.5
        return sigma
    
    def synchrotron_tune(self, Vrf):
        """
        Compute synchrotron tune from RF voltage
        
        Parameters
        ----------
        Vrf : float
            Main RF voltage in [V].
            
        Returns
        -------
        tuneS : float
            Synchrotron tune.
        """
        phase = np.pi - np.arcsin(self.U0 / Vrf)
        tuneS = np.sqrt( - (Vrf / self.E0) * (self.h * self.ac) / (2*np.pi) 
                        * np.cos(phase) )
        return tuneS
