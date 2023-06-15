# -*- coding: utf-8 -*-
"""
Define coherent synchrotron radiation (CSR) wakefields in various models.
"""
import numpy as np
import mpmath as mp
from scipy.constants import c, mu_0
from scipy.special import gamma
from mbtrack2_cuda.impedance.wakefield import WakeField, Impedance, WakeFunction

class FreeSpaceCSR(WakeField):
    """
    Free space steady-state coherent synchrotron radiation Wakefield element,
    based on [1]. 
    
    Impedance is computed using Eq. (A10) of [2].

    Parameters
    ----------
    time : array of float
        Time points where the wake function will be evaluated in [s].
    frequency : array of float
        Frequency points where the impedance will be evaluated in [Hz].
    length : float 
        Length of the impedacen to consider in [m].
    radius : float 
        Dipole radius of curvature in [m].
        
    References
    ----------
    [1] : Faltens, A. N. D. R. I. S., & Laslett, L. J. (1973). Longitudinal 
    coupling impedance of a stationary electron ring in a cylindrical 
    geometry. part. Accel., 4, 151-157.
    [2] : Agoh, T., and K. Yokoya. "Calculation of coherent synchrotron 
    radiation using mesh." Physical Review Special Topics-Accelerators and 
    Beams 7.5 (2004): 054403.

    """
    
    def __init__(self, time, frequency, length, radius):
        super().__init__()
        
        self.length = length
        self.radius = radius
        self.Z0 = mu_0*c

        Zl = self.LongitudinalImpedance(frequency)
        # Wl = self.LongitudinalWakeFunction(time)
        
        Zlong = Impedance(variable = frequency, function = Zl, component_type='long')
        # Wlong = WakeFunction(variable = time, function = Wl, component_type="long")
        
        super().append_to_model(Zlong)
        # super().append_to_model(Wlong)
        
    def LongitudinalImpedance(self, frequency):
        """
        Compute the free space steady-state CSR impedance.
        Based on Eq. (A10) of [1].
        
        This formula is valid only if omega << (3 * gamma^3 * c) / (2 * R).

        Parameters
        ----------
        frequency : float array
            Frequency in [Hz].

        Returns
        -------
        Zl : complex array
            Longitudinal impedance in [ohm].
            
        References
        ----------
        [1] : Agoh, T., and K. Yokoya. "Calculation of coherent synchrotron 
        radiation using mesh." Physical Review Special Topics-Accelerators and 
        Beams 7.5 (2004): 054403.
        """
        
        Zl = (self.Z0 * self.length / (2*np.pi) * gamma(2/3) * 
              (-1j * 2 * np.pi * frequency / (3 * c * self.radius**2) )**(1/3) )
        return Zl
    
    def LongitudinalWakeFunction(self, time):
        raise NotImplementedError
        
class ParallelPlatesCSR(WakeField):
    """
    Perfectly conducting parallel plates steady-state coherent synchrotron 
    radiation Wakefield element, based on [1]. 

    Parameters
    ----------
    time : array of float
        Time points where the wake function will be evaluated in [s].
    frequency : array of float
        Frequency points where the impedance will be evaluated in [Hz].
    length : float 
        Length of the impedacen to consider in [m].
    radius : float 
        Dipole radius of curvature in [m].
    distance : float
        Vertical distance between the parallel plates in [m].
        
    Attributes
    ----------
    threshold : float
        Shielding threshold in the parallel plates model in [Hz].
        
    References
    ----------
    [1] : Agoh, T., and K. Yokoya. "Calculation of coherent synchrotron 
    radiation using mesh." Physical Review Special Topics-Accelerators and 
    Beams 7.5 (2004): 054403.

    """
    
    def __init__(self, time, frequency, length, radius, distance):
        super().__init__()
        
        self.length = length
        self.radius = radius
        self.distance = distance
        self.Z0 = mu_0*c

        Zl = self.LongitudinalImpedance(frequency)
        # Wl = self.LongitudinalWakeFunction(time)
        
        Zlong = Impedance(variable = frequency, function = Zl, component_type='long')
        # Wlong = WakeFunction(variable = time, function = Wl, component_type="long")
        
        super().append_to_model(Zlong)
        # super().append_to_model(Wlong)
        
    @property
    def threshold(self):
        """Shielding threshold in the parallel plates model in [Hz]."""
        return (3 * c) / (2 * np.pi) * (self.radius / self.distance ** 3) ** 0.5
        
    def LongitudinalImpedance(self, frequency, tol=1e-5):
        """
        Compute the CSR impedance using the perfectly conducting parallel 
        plates steady-state model.
        
        Impedance is computed using Eq. (A1) of [1].

        Parameters
        ----------
        frequency : float array
            Frequency in [Hz].
        tol : float, optinal
            Desired maximum final error on sum_func. 

        Returns
        -------
        Zl : complex array
            Longitudinal impedance in [ohm].
            
        References
        ----------
        [1] : Agoh, T., and K. Yokoya. "Calculation of coherent synchrotron 
        radiation using mesh." Physical Review Special Topics-Accelerators and 
        Beams 7.5 (2004): 054403.
        """
        
        Zl = np.zeros(frequency.shape, dtype=complex)
        constant = (2 * np.pi * self.Z0* self.length / self.distance 
                    * (2 / self.radius)**(1/3) )
        for i, f in enumerate(frequency):
            k = 2 * mp.pi * f / c
            
            sum_value = mp.nsum(lambda p: self.sum_func(p, k), [0,mp.inf], 
                                tol=tol, method='r+s')
            
            Zl[i] = constant * (1/k)**(1/3) * complex(sum_value)
    
        return Zl
    
    def sum_func(self, p, k):
        """
        Utility function for LongitudinalImpedance.

        Parameters
        ----------
        p : int
        k : float

        Returns
        -------
        sum_value : mpc

        """
        xp = (2*p + 1)*mp.pi / self.distance * ( self.radius / 2 / k**2 )**(1/3)
        Ai = mp.airyai(xp**2)
        Bi = mp.airybi(xp**2)
        Aip = mp.airyai(xp**2,1)
        Bip = mp.airybi(xp**2,1)
        return Aip*(Aip + 1j*Bip) + xp**2 * Ai * (Ai + 1j*Bi)
    
    def LongitudinalWakeFunction(self, time):
        raise NotImplementedError
