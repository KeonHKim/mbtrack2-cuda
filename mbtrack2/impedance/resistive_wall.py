# -*- coding: utf-8 -*-
"""
Define resistive wall elements based on the WakeField class.
"""

import numpy as np
from scipy.constants import mu_0, epsilon_0, c
from scipy.integrate import quad
from mbtrack2.impedance.wakefield import WakeField, Impedance, WakeFunction

def skin_depth(frequency, rho, mu_r = 1, epsilon_r = 1):
    """
    General formula for the skin depth.
    
    Parameters
    ----------
    frequency : array of float
        Frequency points in [Hz].
    rho : float
        Resistivity in [ohm.m].
    mu_r : float, optional
        Relative magnetic permeability.
    epsilon_r : float, optional
        Relative electric permittivity.
    
    Returns
    -------
    delta : array of float
        Skin depth in [m].
    
    """
    
    delta = (np.sqrt(2*rho/(np.abs(2*np.pi*frequency)*mu_r*mu_0)) * 
             np.sqrt(np.sqrt(1 + (rho*np.abs(2*np.pi*frequency) * 
                                  epsilon_r*epsilon_0)**2 ) 
                    + rho*np.abs(2*np.pi*frequency)*epsilon_r*epsilon_0))
    return delta
    

class CircularResistiveWall(WakeField):
    """
    Resistive wall WakeField element for a circular beam pipe.
    
    Impedance from approximated formulas from Eq. (2.77) of Chao book [1].
    Wake function formulas from [2].
        
    Parameters
    ----------
    time : array of float
        Time points where the wake function will be evaluated in [s].
    frequency : array of float
        Frequency points where the impedance will be evaluated in [Hz].
    length : float 
        Beam pipe length in [m].
    rho : float
        Resistivity in [ohm.m].
    radius : float 
        Beam pipe radius in [m].
    exact : bool, optional
        If False, approxmiated formulas are used for the wake function 
        computations.
    atol : float, optional
        Absolute tolerance used to enforce fundamental theorem of beam loading
        for the exact expression of the longitudinal wake function.
        Default is 1e-20.
    
    References
    ----------
    [1] : Chao, A. W. (1993). Physics of collective beam instabilities in high 
    energy accelerators. Wiley.
    [2] : Skripka, Galina, et al. "Simultaneous computation of intrabunch and 
    interbunch collective beam motions in storage rings." Nuclear Instruments 
    and Methods in Physics Research Section A: Accelerators, Spectrometers, 
    Detectors and Associated Equipment 806 (2016): 221-230.

    """

    def __init__(self, time, frequency, length, rho, radius, exact=False, 
                 atol=1e-20):
        super().__init__()
        
        self.length = length
        self.rho = rho
        self.radius = radius
        self.Z0 = mu_0*c
        self.t0 = (2*self.rho*self.radius**2 / self.Z0)**(1/3) / c
        
        omega = 2*np.pi*frequency
        Z1 = length*(1 + np.sign(frequency)*1j)*rho/(
                2*np.pi*radius*skin_depth(frequency,rho))
        Z2 = c/omega*length*(1 + np.sign(frequency)*1j)*rho/(
                np.pi*radius**3*skin_depth(frequency,rho))
        
        Wl = self.LongitudinalWakeFunction(time, exact, atol)
        Wt = self.TransverseWakeFunction(time, exact)
        
        Zlong = Impedance(variable = frequency, function = Z1, component_type='long')
        Zxdip = Impedance(variable = frequency, function = Z2, component_type='xdip')
        Zydip = Impedance(variable = frequency, function = Z2, component_type='ydip')
        Wlong = WakeFunction(variable = time, function = Wl, component_type="long")
        Wxdip = WakeFunction(variable = time, function = Wt, component_type="xdip")
        Wydip = WakeFunction(variable = time, function = Wt, component_type="ydip")
        
        super().append_to_model(Zlong)
        super().append_to_model(Zxdip)
        super().append_to_model(Zydip)
        super().append_to_model(Wlong)
        super().append_to_model(Wxdip)
        super().append_to_model(Wydip)
        
    def LongitudinalWakeFunction(self, time, exact=False, atol=1e-20):
        """
        Compute the longitudinal wake function of a circular resistive wall 
        using Eq. (22), or approxmiated expression Eq. (24), of [1]. The 
        approxmiated expression is valid if the time is large compared to the 
        characteristic time t0.
        
        If some time value is smaller than atol, then the fundamental theorem 
        of beam loading is applied: Wl(0) = Wl(0+)/2.

        Parameters
        ----------
        time : array of float
            Time points where the wake function is evaluated in [s].
        exact : bool, optional
            If True, the exact expression is used. The default is False.
        atol : float, optional
            Absolute tolerance used to enforce fundamental theorem of beam loading
            for the exact expression of the longitudinal wake function.
            Default is 1e-20.

        Returns
        -------
        wl : array of float
            Longitudinal wake function in [V/C].
            
        References
        ----------
        [1] : Skripka, Galina, et al. "Simultaneous computation of intrabunch and 
        interbunch collective beam motions in storage rings." Nuclear Instruments 
        and Methods in Physics Research Section A: Accelerators, Spectrometers, 
        Detectors and Associated Equipment 806 (2016): 221-230.
        """
        wl = np.zeros_like(time)
        idx1 = time < 0
        wl[idx1] = 0
        if exact==True:
            idx2 = time > 20 * self.t0
            idx3 = np.logical_not(np.logical_or(idx1,idx2))
            wl[idx3] = self.__LongWakeExact(time[idx3], atol)
        else:
            idx2 = np.logical_not(idx1)
        wl[idx2] = self.__LongWakeApprox(time[idx2])
        return wl
    
    def TransverseWakeFunction(self, time, exact=False):
        """
        Compute the transverse wake function of a circular resistive wall 
        using Eq. (25), or approxmiated expression Eq. (26), of [1]. The 
        approxmiated expression is valid if the time is large compared to the 
        characteristic time t0.
        
        Exact expression (Eq. (25) from [1]) is corrected by factor (c * t0).

        Parameters
        ----------
        time : array of float
            Time points where the wake function is evaluated in [s].
        exact : bool, optional
            If True, the exact expression is used. The default is False.

        Returns
        -------
        wt : array of float
            Transverse wake function in [V/C].
            
        References
        ----------
        [1] : Skripka, Galina, et al. "Simultaneous computation of intrabunch and 
        interbunch collective beam motions in storage rings." Nuclear Instruments 
        and Methods in Physics Research Section A: Accelerators, Spectrometers, 
        Detectors and Associated Equipment 806 (2016): 221-230.
        """
        wt = np.zeros_like(time)
        idx1 = time < 0
        wt[idx1] = 0
        if exact==True:
            idx2 = time > 20 * self.t0
            idx3 = np.logical_not(np.logical_or(idx1,idx2))
            wt[idx3] = self.__TransWakeExact(time[idx3])
        else:
            idx2 = np.logical_not(idx1)
        wt[idx2] = self.__TransWakeApprox(time[idx2])
        return wt
    
    def __LongWakeExact(self, time, atol):
        wl = np.zeros_like(time)
        factor = 4*self.Z0*c/(np.pi * self.radius**2) * self.length
        for i, t in enumerate(time):
            val, err = quad(lambda z:self.__function(t, z), 0, np.inf)
            wl[i] = factor * ( np.exp(-t/self.t0) / 3 * 
                              np.cos( np.sqrt(3) * t / self.t0 )  
                              - np.sqrt(2) / np.pi * val )
            if np.isclose(0, t, atol=atol):
                wl[i] = wl[i]/2
        return wl
    
    def __TransWakeExact(self, time):
        wt = np.zeros_like(time)
        factor = ((8 * self.Z0 * c**2 * self.t0) / (np.pi * self.radius**4) * 
                  self.length)
        for i, t in enumerate(time):
            val, err = quad(lambda z:self.__function2(t, z), 0, np.inf)
            wt[i] = factor * ( 1 / 12 * (-1 * np.exp(-t/self.t0) * 
                                      np.cos( np.sqrt(3) * t / self.t0 ) + 
                                      np.sqrt(3) * np.exp(-t/self.t0) * 
                                      np.sin( np.sqrt(3) * t / self.t0 ) ) -
                                      np.sqrt(2) / np.pi * val )
        return wt
    
    def __LongWakeApprox(self, t):
        wl = - 1 * ( 1 / (4*np.pi * self.radius) * 
                    np.sqrt(self.Z0 * self.rho / (c * np.pi) ) /
                    t ** (3/2) ) * self.length
        return wl
    
    def __TransWakeApprox(self, t):
        wt = (1 / (np.pi * self.radius**3) *
              np.sqrt(self.Z0 * c * self.rho / np.pi)
              / t ** (1/2) * self.length)
        return wt
    
    def __function(self, t, x):
        return ( (x**2 * np.exp(-1* (x**2) * t / self.t0) ) / (x**6 + 8) )
    
    def __function2(self, t, x):
        return ( (-1 * np.exp(-1* (x**2) * t / self.t0) ) / (x**6 + 8) )
    
class Coating(WakeField):
    
    def __init__(self, frequency, length, rho1, rho2, radius, thickness, approx=False):
        """
        WakeField element for a coated circular beam pipe.
        
        The longitudinal and tranverse impedances are computed using formulas
        from [1].

        Parameters
        ----------
        f : array of float
            Frequency points where the impedance is evaluated in [Hz].
        length : float
            Length of the beam pipe to consider in [m].
        rho1 : float
            Resistivity of the coating in [ohm.m].
        rho2 : float
            Resistivity of the bulk material in [ohm.m].
        radius : float
            Radius of the beam pipe to consier in [m].
        thickness : float
            Thickness of the coating in [m].
        approx : bool, optional
            If True, used approxmiated formula. The default is False.

        References
        ----------
        [1] : Migliorati, M., E. Belli, and M. Zobov. "Impact of the resistive 
        wall impedance on beam dynamics in the Future Circular e+ e− Collider." 
        Physical Review Accelerators and Beams 21.4 (2018): 041001.

        """
        super().__init__()
        
        self.length = length
        self.rho1 = rho1
        self.rho2 = rho2
        self.radius = radius
        self.thickness = thickness
        
        Zl = self.LongitudinalImpedance(frequency, approx)
        Zt = self.TransverseImpedance(frequency, approx)
        
        Zlong = Impedance(variable = frequency, function = Zl, component_type='long')
        Zxdip = Impedance(variable = frequency, function = Zt, component_type='xdip')
        Zydip = Impedance(variable = frequency, function = Zt, component_type='ydip')
        
        super().append_to_model(Zlong)
        super().append_to_model(Zxdip)
        super().append_to_model(Zydip)
        
    def LongitudinalImpedance(self, f, approx):
        """
        Compute the longitudinal impedance of a coating using Eq. (5), or 
        approxmiated expression Eq. (8), of [1]. The approxmiated expression 
        is valid if the skin depth of the coating is large compared to the 
        coating thickness. 

        Parameters
        ----------
        f : array of float
            Frequency points where the impedance is evaluated in [Hz].
        approx : bool
            If True, used approxmiated formula.

        Returns
        -------
        Zl : array
            Longitudinal impedance values in [ohm].
            
        References
        ----------
        [1] : Migliorati, M., E. Belli, and M. Zobov. "Impact of the resistive 
        wall impedance on beam dynamics in the Future Circular e+ e− Collider." 
        Physical Review Accelerators and Beams 21.4 (2018): 041001.

        """
        
        Z0 = mu_0*c
        factor = Z0*f/(2*c*self.radius)*self.length
        skin1 = skin_depth(f, self.rho1)
        skin2 = skin_depth(f, self.rho2)
        
        if approx == False:
            alpha = skin1/skin2
            tanh = np.tanh( (1 + 1j*np.sign(f)) * self.thickness / skin1 )
            bracket = ( (np.sign(f) + 1j) * skin1 * 
                       (alpha * tanh + 1) / (alpha + tanh) )
        else:
            valid_approx = self.thickness / np.min(skin1)
            if valid_approx < 0.01:
                print("Approximation is not valid. Returning impedance anyway.")
            bracket = ( (np.sign(f) + 1j) * skin2 + 2 * 1j * self.thickness * 
                       (1 - self.rho2/self.rho1) )
        
        Zl = factor * bracket
        
        return Zl
        
    def TransverseImpedance(self, f, approx):
        """
        Compute the transverse impedance of a coating using Eq. (6), or 
        approxmiated expression Eq. (9), of [1]. The approxmiated expression 
        is valid if the skin depth of the coating is large compared to the 
        coating thickness. 

        Parameters
        ----------
        f : array of float
            Frequency points where the impedance is evaluated in [Hz].
        approx : bool
            If True, used approxmiated formula.

        Returns
        -------
        Zt : array
            Transverse impedance values in [ohm].
            
        References
        ----------
        [1] : Migliorati, M., E. Belli, and M. Zobov. "Impact of the resistive 
        wall impedance on beam dynamics in the Future Circular e+ e− Collider." 
        Physical Review Accelerators and Beams 21.4 (2018): 041001.

        """
        
        Z0 = mu_0*c
        factor = Z0/(2*np.pi*self.radius**3)*self.length
        skin1 = skin_depth(f, self.rho1)
        skin2 = skin_depth(f, self.rho2)
        
        if approx == False:
            alpha = skin1/skin2
            tanh = np.tanh( (1 + 1j*np.sign(f)) * self.thickness / skin1 )
            bracket = ( (1 + 1j*np.sign(f)) * skin1 * 
                       (alpha * tanh + 1) / (alpha + tanh) )
        else:
            valid_approx = self.thickness / np.min(skin1)
            if valid_approx < 0.01:
                print("Approximation is not valid. Returning impedance anyway.")
            bracket = ( (1 + 1j*np.sign(f)) * skin2 + 2 * 1j * self.thickness 
                       * np.sign(f) * (1 - self.rho2/self.rho1) )
        
        Zt = factor * bracket
        
        return Zt
