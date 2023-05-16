# -*- coding: utf-8 -*-
"""
This module defines the impedances and wake functions from the resonator model 
based on the WakeField class.
"""

import numpy as np
from mbtrack2_cuda.impedance.wakefield import (WakeField, Impedance, 
                                                   WakeFunction)

class Resonator(WakeField):
    def __init__(self, time, frequency, Rs, fr, Q, plane, atol=1e-20):
        """
        Resonator model WakeField element which computes the impedance and the 
        wake function in both longitudinal and transverse case.

        Parameters
        ----------
        time : array of float
            Time points where the wake function will be evaluated in [s].
        frequency : array of float
            Frequency points where the impedance will be evaluated in [Hz].
        Rs : float
            Shunt impedance in [ohm].
        fr : float
            Resonance frequency in [Hz].
        Q : float
            Quality factor.
        plane : str or list
            Plane on which the resonator is used: "long", "x" or "y".
        atol : float, optional
            Absolute tolerance used to enforce fundamental theorem of beam 
            loading for the exact expression of the longitudinal wake function.
            Default is 1e-20.
            
        References
        ----------
        [1]  B. W. Zotter and S. A. Kheifets, "Impedances and Wakes in High-Energy 
        Particle Ac-celerators", Eq. (3.10) and (3.15), pp.51-53.

        """
        super().__init__()
        self.Rs = Rs
        self.fr = fr
        self.wr = 2 * np.pi * self.fr
        self.Q = Q
        if isinstance(plane, str):
            self.plane = [plane]
        elif isinstance(plane, list):
            self.plane = plane
            
        if self.Q >= 0.5:
            self.Q_p = np.sqrt(self.Q**2 - 0.25)
        else:
            self.Q_p = np.sqrt(0.25 - self.Q**2)
        self.wr_p = (self.wr*self.Q_p)/self.Q
        
        for dim in self.plane:
            if dim == "long":
                Zlong = Impedance(variable=frequency, 
                                function=self.long_impedance(frequency),
                                component_type="long")
                super().append_to_model(Zlong)
                Wlong = WakeFunction(variable=time,
                                    function=self.long_wake_function(time, atol),
                                    component_type="long")
                super().append_to_model(Wlong)
                
            elif dim == "x" or dim == "y":
                Zdip = Impedance(variable=frequency, 
                                function=self.transverse_impedance(frequency),
                                component_type=dim + "dip")
                super().append_to_model(Zdip)
                Wdip = WakeFunction(variable=time,
                                    function=self.transverse_wake_function(time),
                                    component_type=dim + "dip")
                super().append_to_model(Wdip)
            else:
                raise ValueError("Plane must be: long, x or y")
        
    def long_wake_function(self, t, atol):
        if self.Q >= 0.5:
            wl = ( (self.wr * self.Rs / self.Q) * 
                    np.exp(-1* self.wr * t / (2 * self.Q) ) *
                     (np.cos(self.wr_p * t) - 
                      np.sin(self.wr_p * t) / (2 * self.Q_p) ) )
        elif self.Q < 0.5:
            wl = ( (self.wr * self.Rs / self.Q) * 
                    np.exp(-1* self.wr * t / (2 * self.Q) ) *
                     (np.cosh(self.wr_p * t) - 
                      np.sinh(self.wr_p * t) / (2 * self.Q_p) ) )
        if np.any(np.abs(t) < atol):
            wl[np.abs(t) < atol] = wl[np.abs(t) < atol]/2
        return wl
                            
    def long_impedance(self, f):
        return self.Rs / (1 + 1j * self.Q * (f/self.fr - self.fr/f))
    
    def transverse_impedance(self, f):
        return self.Rs * self.fr / f / (
            1 + 1j * self.Q * (f / self.fr - self.fr / f) )
    
    def transverse_wake_function(self, t):
        if self.Q >= 0.5:
            return (self.wr * self.Rs / self.Q_p * 
                    np.exp(-1 * t * self.wr / 2 / self.Q_p) *
                    np.sin(self.wr_p * t) )
        else:
            return (self.wr * self.Rs / self.Q_p * 
                    np.exp(-1 * t * self.wr / 2 / self.Q_p) *
                    np.sinh(self.wr_p * t) )
    
class PureInductive(WakeField):
    """
    Pure inductive Wakefield element which computes associated longitudinal 
    impedance and wake function.
    
    Parameters
    ----------
    L : float
        Inductance value in [Ohm/Hz].
    n_wake : int or float, optional
        Number of points used in the wake function.
    n_imp : int or float, optional
        Number of points used in the impedance.
    imp_freq_lim : float, optional
        Maximum frequency used in the impedance. 
    nout, trim : see Impedance.to_wakefunction
    """
    def __init__(self, L, n_wake=1e6, n_imp=1e6, imp_freq_lim=1e11, nout=None,
                 trim=False):
        self.L = L
        self.n_wake = int(n_wake)
        self.n_imp = int(n_imp)
        self.imp_freq_lim = imp_freq_lim
        
        freq = np.linspace(start=1, stop=self.imp_freq_lim, num=self.n_imp)
        imp = Impedance(variable=freq, 
                        function=self.long_impedance(freq),
                        component_type="long")
        super().append_to_model(imp)
        
        wf = imp.to_wakefunction(nout=nout, trim=trim)
        super().append_to_model(wf)
        
    def long_impedance(self, f):
        return 1j*self.L*f
    
class PureResistive(WakeField):
    """
    Pure resistive Wakefield element which computes associated longitudinal 
    impedance and wake function.
    
    Parameters
    ----------
    R : float
        Resistance value in [Ohm].
    n_wake : int or float, optional
        Number of points used in the wake function.
    n_imp : int or float, optional
        Number of points used in the impedance.
    imp_freq_lim : float, optional
        Maximum frequency used in the impedance. 
    nout, trim : see Impedance.to_wakefunction
    """
    def __init__(self, R, n_wake=1e6, n_imp=1e6, imp_freq_lim=1e11, nout=None,
                 trim=False):
        self.R = R
        self.n_wake = int(n_wake)
        self.n_imp = int(n_imp)
        self.imp_freq_lim = imp_freq_lim
        
        freq = np.linspace(start=1, stop=self.imp_freq_lim, num=self.n_imp)
        imp = Impedance(variable=freq, 
                        function=self.long_impedance(freq),
                        component_type="long")
        super().append_to_model(imp)
        
        wf = imp.to_wakefunction(nout=nout, trim=trim)
        super().append_to_model(wf)
        
    def long_impedance(self, f):
        return self.R