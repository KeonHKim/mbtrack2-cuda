# -*- coding: utf-8 -*-
"""
Various calculations about ion trapping and instabilities in electron storage 
rings.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, m_e, e, pi, epsilon_0, hbar, Boltzmann, physical_constants, m_p

rp = 1/(4*pi*epsilon_0) * e**2 / (m_p * c**2)
re = physical_constants["classical electron radius"][0]

def ion_cross_section(ring, ion):
    """
    Compute the collisional ionization cross section.
    
    Compute the inelastic collision cross section between a molecule or an atom
    by a relativistic electron using the relativistic Bethe asymptotic formula 
    [1].
    
    Values of M02 and C02 from [2].

    Parameters
    ----------
    ring : Synchrotron object
    ion : str
        Ion type.

    Returns
    -------
    sigma : float
        Cross section in [m**2].
        
    References
    ----------
    [1] : M. Inokuti, "Inelastic collisions of fast charged particles with 
    atoms and molecules-the bethe theory revisited", Reviews of modern physics 
    43 (1971).
    [2] : P. F. Tavares, "Bremsstrahlung detection of ions trapped in the EPA 
    electron beam", Part. Accel. 43 (1993).

    """
    if ion == "CO":
        M02 = 3.7
        C0 = 35.1
    elif ion == "H2":
        M02 = 0.7
        C0 = 8.1
    else:
        raise NotImplementedError
        
    sigma = 4*pi*(hbar/m_e/c)**2 * (M02 *(1/ring.beta**2 * np.log(ring.beta**2/(1-ring.beta**2)) - 1) + C0/ring.beta**2)
    
    return sigma

def ion_frequency(N, Lsep, sigmax, sigmay, ion="CO", dim="y", express="coupling"):
    """
    Compute the ion oscillation frequnecy.

    Parameters
    ----------
    N : float
        Number of electrons per bunch.
    Lsep : float
        Bunch spacing in [m].
    sigmax : float or array
        Horizontal beam size in [m].
    sigmay : float or array
        Vertical beam size in [m].
    ion : str, optional
        Ion type. The default is "CO".
    dim : "y" o "x", optional
        Dimension to consider. The default is "y".
    express : str, optional
        Expression to use to compute the ion oscillation frequency. 
        The default is "coupling" corresponding to Gaussian electron and ion 
        distributions with coupling [1].
        Also possible is "no_coupling" corresponding to Gaussian electron and 
        ion distributions without coupling [2].

    Returns
    -------
    f : float or array
        Ion oscillation frequencies in [Hz].
        
    References
    ----------
    [1] : T. O. Raubenheimer and F. Zimmermann, "Fast beam-ion instability. I. 
    linear theory and simulations", Physical Review E 52 (1995).
    [2] : G. V. Stupakov, T. O. Raubenheimer, and F. Zimmermann, "Fast beam-ion 
    instability. II. effect of ion decoherence", Physical Review E 52 (1995).

    """
    
    if ion == "CO":
        A = 28
    elif ion == "H2":
        A = 2
    elif ion == "CH4":
        A = 18
    elif ion == "H2O":
        A = 16
    elif ion == "CO2":
        A = 44
    
    if dim == "y":
        pass
    elif dim == "x":
        sigmay, sigmax = sigmax, sigmay
    else:
        raise ValueError
        
    if express == "coupling":
        k = 3/2
    elif express == "no_coupling":
        k = 1
        
    f = c * np.sqrt( 2 * rp * N / ( A * k * Lsep * sigmay * (sigmax + sigmay) ) ) / (2*pi)
    
    return f
    
def fast_beam_ion(ring, Nb, nb, Lsep, sigmax, sigmay, P, T, beta, 
                  model="linear", delta_omega = 0, ion="CO", dim="y"):
    """
    Compute fast beam ion instability rise time [1].
    
    Warning ! 
    If model="linear", the rise time is an assymptotic grow time 
    (i.e. y ~ exp(sqrt(t/tau))) [1].
    If model="decoherence", the rise time is an e-folding time 
    (i.e. y ~ exp(t/tau)) [2].
    If model="non-linear", the rise time is a linear growth time
    (i.e. y ~ t/tau) [3].
    
    The linear model assumes that [1]:
        x,y << sigmax,sigmay
    
    The decoherence model assumes that [2]:
        Lsep << c / (2 * pi * ion_frequency) 
        Lsep << c / (2 * pi * betatron_frequency)
        
    The non-linear model assumes that [3]:
        x,y >> sigmax,sigmay

    Parameters
    ----------
    ring : Synchrotron object
    Nb : float
        Number of electron per bunch.
    nb : float
        Number of bunches.
    Lsep : float
        Bunch spacing in [m].
    sigmax : float
        Horizontal beam size in [m].
    sigmay : float
        Vertical beam size in [m].
    P : float
        Partial pressure of the molecular ion in [Pa].
    T : float
        Tempertature in [K].
    beta : float
        Average betatron function around the ring in [m].
    model : str, optional
        If "linear", use [1].
        If "decoherence", use [2].
        If "non-linear", use [3].
    delta_omega : float, optional
        RMS variation of the ion oscillation angular frequnecy around the ring
        in [Hz].
    ion : str, optional
        Ion type. The default is "CO".
    dim : "y" o "x", optional
        Dimension to consider. The default is "y".

    Returns
    -------
    tau : float
        Instability rise time in [s].
        
    References
    ----------
    [1] : T. O. Raubenheimer and F. Zimmermann, "Fast beam-ion instability. I. 
    linear theory and simulations", Physical Review E 52 (1995).
    [2] : G. V. Stupakov, T. O. Raubenheimer, and F. Zimmermann, "Fast beam-ion 
    instability. II. effect of ion decoherence", Physical Review E 52 (1995).
    [3] : Chao, A. W., & Mess, K. H. (Eds.). (2013). Handbook of accelerator 
    physics and engineering. World scientific. 3rd Printing. p417.
    
    """
    if dim == "y":
        pass
    elif dim == "x":
        sigmay, sigmax = sigmax, sigmay
    else:
        raise ValueError
        
    if ion == "CO":
        A = 28
    elif ion == "H2":
        A = 2
        
    sigma_i = ion_cross_section(ring, ion)
    
    d_gas = P/(Boltzmann*T)
    
    num = 4 * d_gas * sigma_i * beta * Nb**(3/2) * nb**2 * re * rp**(1/2) * Lsep**(1/2) * c
    den = 3 * np.sqrt(3) * ring.gamma * sigmay**(3/2) * (sigmay + sigmax)**(3/2) * A**(1/2)
    
    tau = den/num
    
    if model == "decoherence":
        tau = tau * 2 * np.sqrt(2) * nb * Lsep * delta_omega / c
    elif model == "non-linear":
        fi = ion_frequency(Nb, Lsep, sigmax, sigmay, ion, dim)
        tau = tau * 2 * pi * fi * ring.T1 * nb**(3/2)
    elif model == "linear":
        pass
    else:
        raise ValueError("model unknown")
    
    return tau 

def plot_critical_mass(ring, bunch_charge, bunch_spacing, n_points=1e4):
    """
    Plot ion critical mass, using Eq. (7.70) p147 of [1]

    Parameters
    ----------
    ring : Synchrotron object
    bunch_charge : float
        Bunch charge in [C].
    bunch_spacing : float
        Time in between two adjacent bunches in [s].
    n_points : float or int, optional
        Number of point used in the plot. The default is 1e4.

    Returns
    -------
    fig : figure
        
    References
    ----------
    [1] : Gamelin, A. (2018). Collective effects in a transient microbunching 
    regime and ion cloud mitigation in ThomX (Doctoral dissertation, 
    Université Paris-Saclay).

    """
    
    n_points = int(n_points)
    s = np.linspace(0, ring.L, n_points)
    sigma = ring.sigma(s)
    N = np.abs(bunch_charge/e)
    
    Ay = N*rp*bunch_spacing*c/(2*sigma[2,:]*(sigma[2,:] + sigma[0,:]))
    Ax = N*rp*bunch_spacing*c/(2*sigma[0,:]*(sigma[2,:] + sigma[0,:]))
    
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(s, Ax, label=r"$A_x^c$")
    ax.plot(s, Ay, label=r"$A_y^c$")
    ax.set_yscale("log")
    ax.plot(s, np.ones_like(s)*2, label=r"$H_2^+$")
    ax.plot(s, np.ones_like(s)*16, label=r"$H_2O^+$")
    ax.plot(s, np.ones_like(s)*18, label=r"$CH_4^+$")
    ax.plot(s, np.ones_like(s)*28, label=r"$CO^+$")
    ax.plot(s, np.ones_like(s)*44, label=r"$CO_2^+$")
    ax.legend()
    ax.set_ylabel("Critical mass")
    ax.set_xlabel("Longitudinal position [m]")
    
    return fig
    