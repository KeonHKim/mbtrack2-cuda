# -*- coding: utf-8 -*-
"""
Module where taper elements are defined.
"""

from scipy.constants import mu_0, c, pi
import numpy as np
from scipy.integrate import trapz
from mbtrack2_cuda.impedance.wakefield import WakeField, Impedance

class StupakovRectangularTaper(WakeField):
    """
    Rectangular vertical taper WakeField element, using the low frequency 
    approxmiation. Assume constant taper angle. Formulas from [1].
    
    ! Valid for low w
    
    Parameters
    ----------
    frequency: frequency points where the impedance will be evaluated in [Hz]
    gap_entrance : full vertical gap at taper entrance in [m]
    gap_exit: full vertical gap at taper exit in [m]
    length: taper length in [m]
    width : full horizontal width of the taper in [m]
    """
    
    def __init__(self, frequency, gap_entrance, gap_exit, length, width, 
                 m_max=100, n_points=int(1e4)):
        super().__init__()
        
        self.frequency = frequency
        self.gap_entrance = gap_entrance
        self.gap_exit = gap_exit
        self.length = length
        self.width = width
        self.m_max = m_max
        self.n_points = n_points

        Zlong = Impedance(variable = frequency, function = self.long(), component_type='long')
        Zxdip = Impedance(variable = frequency, function = self.xdip(), component_type='xdip')
        Zydip = Impedance(variable = frequency, function = self.ydip(), component_type='ydip')
        Zxquad = Impedance(variable = frequency, function = -1*self.quad(), component_type='xquad')
        Zyquad = Impedance(variable = frequency, function = self.quad(), component_type='yquad')
        
        super().append_to_model(Zlong)
        super().append_to_model(Zxdip)
        super().append_to_model(Zydip)
        super().append_to_model(Zxquad)
        super().append_to_model(Zyquad)
        
    @property
    def gap_prime(self):
        return (self.gap_entrance-self.gap_exit)/self.length
    
    @property
    def angle(self):
        return np.arctan((self.gap_entrance/2 - self.gap_exit/2)/self.length)
    
    @property
    def Z0(self):
        return mu_0*c

    def long(self, frequency=None):
        
        if frequency is None:
            frequency = self.frequency
        
        def F(x, m_max):
            m = np.arange(0, m_max)
            phi = np.outer(pi*x/2, 2*m+1)
            val = 1/(2*m+1)/(np.cosh(phi)**2)*np.tanh(phi)
            return val.sum(1)
    
        z = np.linspace(0, self.length, self.n_points)
        g = np.linspace(self.gap_entrance, self.gap_exit, self.n_points)
        
        to_integrate = self.gap_prime**2*F(g/self.width, self.m_max)
        integral = trapz(to_integrate,x=z)
        
        return -1j*frequency*self.Z0/(2*c)*integral
    
    def Z_over_n(self, f0):
        return np.imag(self.long(1))*f0

    def ydip(self):
        
        def G1(x, m_max):
            m = np.arange(0, m_max)
            phi = np.outer(pi*x/2, 2*m+1)
            val = (2*m+1)/(np.sinh(phi)**2)/np.tanh(phi)
            val = x[:,None]**3*val
            return val.sum(1)
        
        z = np.linspace(0, self.length, self.n_points)
        g = np.linspace(self.gap_entrance, self.gap_exit, self.n_points)
        
        to_integrate = self.gap_prime**2/(g**3)*G1(g/self.width, self.m_max)
        integral = trapz(to_integrate, x=z)
        
        return -1j*pi*self.width*self.Z0/4*integral
    
    def xdip(self):
        
        def G3(x, m_max):
            m = np.arange(0,m_max)
            phi = np.outer(pi*x, m)
            val = 2*m/(np.cosh(phi)**2)*np.tanh(phi)
            val = x[:,None]**2*val
            return val.sum(1)
        
        z = np.linspace(0, self.length, self.n_points)
        g = np.linspace(self.gap_entrance, self.gap_exit, self.n_points)
        
        to_integrate = self.gap_prime**2/(g**2)*G3(g/self.width, self.m_max)
        integral = trapz(to_integrate, x=z)
        
        return -1j*pi*self.Z0/4*integral
    
    
    def quad(self):
        
        def G2(x, m_max):
            m = np.arange(0, m_max)
            phi = np.outer(pi*x/2, 2*m+1)
            val = (2*m+1)/(np.cosh(phi)**2)*np.tanh(phi)
            val = x[:,None]**2*val
            return val.sum(1)
    
        z = np.linspace(0, self.length, self.n_points)
        g = np.linspace(self.gap_entrance, self.gap_exit, self.n_points)
        
        to_integrate = self.gap_prime**2/(g**2)*G2(g/self.width, self.m_max)
        integral = trapz(to_integrate, x=z)
        
        return -1j*pi*self.Z0/4*integral
    
class StupakovCircularTaper(WakeField):
    """
    Circular taper WakeField element, using the low frequency 
    approxmiation. Assume constant taper angle. Formulas from [1].
    
    ! Valid for low w
    
    Parameters
    ----------
    frequency: frequency points where the impedance will be evaluated in [Hz]
    radius_entrance : radius at taper entrance in [m]
    radius_exit : radius at taper exit in [m]
    length : taper length in [m]
    """
    
    def __init__(self, frequency, radius_entrance, radius_exit, length,
                 m_max=100, n_points=int(1e4)):
        super().__init__()
        
        self.frequency = frequency
        self.radius_entrance = radius_entrance
        self.radius_exit = radius_exit
        self.length = length
        self.m_max = m_max
        self.n_points = n_points

        Zlong = Impedance(variable = frequency, function = self.long(), component_type='long')
        Zxdip = Impedance(variable = frequency, function = self.dip(), component_type='xdip')
        Zydip = Impedance(variable = frequency, function = self.dip(), component_type='ydip')
        
        super().append_to_model(Zlong)
        super().append_to_model(Zxdip)
        super().append_to_model(Zydip)
        
    @property
    def angle(self):
        return np.arctan((self.radius_entrance-self.radius_exit)/self.length)
    
    @property
    def radius_prime(self):
        return (self.radius_entrance-self.radius_exit)/self.length
    
    @property
    def Z0(self):
        return mu_0*c

    def long(self, frequency=None):
        
        if frequency is None:
            frequency = self.frequency
        
        return (self.Z0/(2*pi)*np.log(self.radius_entrance/self.radius_exit) 
                - 1j*self.Z0*frequency/(2*c)*self.radius_prime**2*self.length)
    
    def Z_over_n(self, f0):
        return np.imag(self.long(1))*f0

    def dip(self, frequency=None):
        
        if frequency is None:
            frequency = self.frequency
        
        z = np.linspace(0, self.length, self.n_points)
        r = np.linspace(self.radius_entrance, self.radius_exit, self.n_points)
        
        to_integrate = self.radius_prime**2/(r**2)
        integral = trapz(to_integrate, x=z)
        
        return (self.Z0*c/(4*pi**2*frequency)*(1/(self.radius_exit**2) - 
               1/(self.radius_entrance**2))  - 1j*self.Z0/(2*pi)*integral)




