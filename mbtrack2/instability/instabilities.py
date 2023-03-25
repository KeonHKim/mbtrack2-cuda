# -*- coding: utf-8 -*-
"""
General calculations about instability thresholds.
"""

import numpy as np
from scipy.constants import c, m_e, e, pi, epsilon_0
import math

def mbi_threshold(ring, sigma, R, b):
    """
    Compute the microbunching instability (MBI) threshold for a bunched beam
    considering the steady-state parallel plate model [1][2].
    
    Parameters
    ----------
    ring : Synchrotron object
    sigma : float
        RMS bunch length in [s]
    R : float
        dipole bending radius in [m]
    b : float
        vertical distance between the conducting parallel plates in [m]
        
    Returns
    -------
    I : float
        MBI current threshold in [A]
        
    [1] : Y. Cai, "Theory of microwave instability and coherent synchrotron 
    radiation in electron storage rings", SLAC-PUB-14561
    [2] : D. Zhou, "Coherent synchrotron radiation and microwave instability in 
    electron storage rings", PhD thesis, p112
    """
    
    sigma = sigma * c
    Ia = 4*pi*epsilon_0*m_e*c**3/e # Alfven current
    chi = sigma*(R/b**3)**(1/2) # Shielding paramter
    xi = 0.5 + 0.34*chi
    N = (ring.L * Ia * ring.ac * ring.gamma * ring.sigma_delta**2 * xi *
        sigma**(1/3) / ( c * e * R**(1/3) ))
    I = N*e/ring.T0
    
    return I

def cbi_threshold(ring, I, Vrf, f, beta, Ncav=1):
    """
    Compute the longitudinal and transverse coupled bunch instability 
    thresolds driven by HOMs [1].
    
    Approximate formula, does not take into account variation with Q.
    For better estimate use lcbi_growth_rate.

    Parameters
    ----------
    ring : Synchrotron object
    I : float
        Total beam current in [A].
    Vrf : float
        Total RF voltage in [V].
    f : float
        Frequency of the HOM in [Hz].
    beta : array-like of shape (2,)
        Horizontal and vertical beta function at the HOM position in [m].
    Ncav : int, optional
        Number of RF cavity.

    Returns
    -------
    Zlong : float
        Maximum longitudinal impedance of the HOM in [ohm].
    Zxdip : float
        Maximum horizontal dipolar impedance of the HOM in [ohm/m].
    Zydip : float
        Maximum vertical dipolar impedance of the HOM in [ohm/m].
        
    References
    ----------
    [1] : Ruprecht, Martin, et al. "Calculation of Transverse Coupled Bunch 
    Instabilities in Electron Storage Rings Driven By Quadrupole Higher Order 
    Modes." 7th Int. Particle Accelerator Conf.(IPAC'16), Busan, Korea. 
    """
    
    fs = ring.synchrotron_tune(Vrf)*ring.f0
    Zlong = fs/(f*ring.ac*ring.tau[2]) * (2*ring.E0) / (ring.f0 * I * Ncav)
    Zxdip = 1/(ring.tau[0]*beta[0]) * (2*ring.E0) / (ring.f0 * I * Ncav)
    Zydip = 1/(ring.tau[1]*beta[1]) * (2*ring.E0) / (ring.f0 * I * Ncav)
    
    return (Zlong, Zxdip, Zydip)

def lcbi_growth_rate_mode(ring, I, Vrf, M, mu, fr=None, Rs=None, QL=None, Z=None):
    """
    Compute the longitudinal coupled bunch instability growth rate driven by
    HOMs for a given coupled bunch mode mu [1].
    
    Use either the resonator model (fr,Rs,QL) or an Impedance object (Z).

    Parameters
    ----------
    ring : Synchrotron object
    I : float
        Total beam current in [A].
    Vrf : float
        Total RF voltage in [V].
    M : int
        Nomber of bunches in the beam.
    mu : int
        Coupled bunch mode number (= 0, ..., M-1).
    fr : float, optional
        Frequency of the HOM in [Hz].
    Rs : float, optional
        Shunt impedance of the HOM in [Ohm].
    QL : float, optional
        Loaded quality factor of the HOM.
    Z : Impedance, optional
        Longitunial impedance to consider.

    Returns
    -------
    float
        Coupled bunch instability growth rate for the mode mu.
        
    References
    ----------
    [1] : Eq. 51 p139 of Akai, Kazunori. "RF System for Electron Storage 
    Rings." Physics And Engineering Of High Performance Electron Storage Rings 
    And Application Of Superconducting Technology. 2002. 118-149.

    """

    nu_s = ring.synchrotron_tune(Vrf)
    factor = ring.eta * I / (4 * np.pi * ring.E0 * nu_s)
    
    if Z is None:
        omega_r = 2 * np.pi * fr
        n_max = int(10 * omega_r / (ring.omega0 * M))
        def Zr(omega):
            return np.real(Rs / (1 + 1j * QL * (omega_r/omega - omega/omega_r)))
    else:
        fmax = Z.data.index.max()
        n_max = int(2 * np.pi * fmax / (ring.omega0 * M))
        def Zr(omega):
            return np.real( Z( omega / (2*np.pi) ) )
        
    n0 = np.arange(n_max)
    n1 = np.arange(1, n_max)
    omega_p = ring.omega0 * (n0 * M + mu + nu_s)
    omega_m = ring.omega0 * (n1 * M - mu - nu_s)
        
    sum_val = np.sum(omega_p*Zr(omega_p)) - np.sum(omega_m*Zr(omega_m))

    return factor * sum_val
    
def lcbi_growth_rate(ring, I, Vrf, M, fr=None, Rs=None, QL=None, Z=None):
    """
    Compute the maximum growth rate for longitudinal coupled bunch instability 
    driven by HOMs [1].
    
    Use either the resonator model (fr,Rs,QL) or an Impedance object (Z).

    Parameters
    ----------
    ring : Synchrotron object
    I : float
        Total beam current in [A].
    Vrf : float
        Total RF voltage in [V].
    M : int
        Nomber of bunches in the beam.
    fr : float, optional
        Frequency of the HOM in [Hz].
    Rs : float, optional
        Shunt impedance of the HOM in [Ohm].
    QL : float, optional
        Loaded quality factor of the HOM.
    Z : Impedance, optional
        Longitunial impedance to consider.

    Returns
    -------
    growth_rate : float
        Maximum coupled bunch instability growth rate.
    mu : int
        Coupled bunch mode number corresponding to the maximum coupled bunch 
        instability growth rate.
    growth_rates : array
        Coupled bunch instability growth rates for the different mode numbers.
        
    References
    ----------
    [1] : Eq. 51 p139 of Akai, Kazunori. "RF System for Electron Storage 
    Rings." Physics And Engineering Of High Performance Electron Storage Rings 
    And Application Of Superconducting Technology. 2002. 118-149.

    """
    growth_rates = np.zeros(M)
    for i in range(M):
        growth_rates[i] = lcbi_growth_rate_mode(ring, I, Vrf, M, i, fr=fr, Rs=Rs, QL=QL, Z=Z)
    
    growth_rate = np.max(growth_rates)
    mu = np.argmax(growth_rates)
    
    return growth_rate, mu, growth_rates
    
def rwmbi_growth_rate(ring, current, beff, rho_material, plane='x'):
    """
    Compute the growth rate of the transverse coupled-bunch instability induced
    by resistive wall impedance [1].

    Parameters
    ----------
    ring : Synchrotron object
    current : float
        Total beam current in [A].
    beff : float
        Effective radius of the vacuum chamber in [m].
    rho_material : float
        Resistivity of the chamber's wall material in [Ohm.m].
    plane : str, optional
        The plane in which the instability will be computed. Use 'x' for the 
        horizontal plane, and 'y' for the vertical.

    Reference
    ---------
    [1] Eq. (31) in R. Nagaoka and K. L. F. Bane, "Collective effects in a
    diffraction-limited storage ring", J. Synchrotron Rad. Vol 21, 2014. pp.937-960 

    """
    plane_dict = {'x':0, 'y':1}
    index = plane_dict[plane]
    beta0 = ring.optics.local_beta[index]
    omega0 = ring.omega0
    E0 = ring.E0
    R = ring.L/(2*np.pi)
    frac_tune, int_tune = math.modf(ring.tune[index])
    Z0 = 377.7 # Vacuum impedance [Ohm]
    
    gr = (beta0*omega0*current*R) /(4*np.pi*E0*beff**3) * ((2*c*Z0*rho_material) / ((1-frac_tune)*omega0))**0.5
    
    return gr

def rwmbi_threshold(ring, beff, rho_material, plane='x'):
    """
    Compute the threshold current of the transverse coupled-bunch instability 
    induced by resistive wall impedance [1].

    Parameters
    ----------
    ring : Synchrotron object
    beff : float
        Effective radius of the vacuum chamber in [m].
    rho_material : float
        Resistivity of the chamber's wall material in [Ohm.m].
    plane : str, optional
        The plane in which the instability will be computed. Use 'x' for the 
        horizontal plane, and 'y' for the vertical.

    Reference
    ---------
    [1] Eq. (32) in R. Nagaoka and K. L. F. Bane, "Collective effects in a
    diffraction-limited storage ring", J. Synchrotron Rad. Vol 21, 2014. pp.937-960 

    """
    plane_dict = {'x':0, 'y':1}
    index = plane_dict[plane]
    beta0 = ring.optics.local_beta[index]
    omega0 = ring.omega0
    E0 = ring.E0
    tau_rad = ring.tau[index]
    frac_tune, int_tune = math.modf(ring.tune[index])
    Z0 = 377.7 # Vacuum impedance [Ohm]
    
    Ith = (4*np.pi*E0*beff**3) / (c*beta0*tau_rad) * (((1-frac_tune)*omega0) / (2*c*Z0*rho_material))**0.5
    
    return Ith
       