# -*- coding: utf-8 -*-
"""
This module defines miscellaneous utilities functions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from mbtrack2_cuda.impedance.wakefield import Impedance
from mbtrack2_cuda.utilities.spectrum import spectral_density
    
def effective_impedance(ring, imp, m, mu, sigma, M, tuneS, xi=None, 
                        mode="Hermite"):
    """
    Compute the effective (longitudinal or transverse) impedance. 
    Formulas from Eq. (1) and (2) p238 of [1].
    
    Parameters
    ----------
    ring : Synchrotron object
    imp : Impedance object
    mu : int
        coupled bunch mode number, goes from 0 to (M-1) where M is the
        number of bunches
    m : int
        head-tail (or azimutal/synchrotron) mode number
    sigma : float
        RMS bunch length in [s]
    M : int
        Number of bunches.
    tuneS : float
        Synchrotron tune.
    xi : float, optional
        (non-normalized) chromaticity
    mode: str, optional
        type of the mode taken into account for the computation:
        -"Hermite" modes for Gaussian bunches

    Returns
    -------
    Zeff : float 
        effective impedance in [ohm] or in [ohm/m] depanding on the impedance
        type.
        
    References
    ----------
    [1] : Handbook of accelerator physics and engineering, 3rd printing.
    """
    
    if not isinstance(imp, Impedance):
        raise TypeError("{} should be an Impedance object.".format(imp))
        
    fmin = imp.data.index.min()
    fmax = imp.data.index.max()
    if fmin > 0:
        double_sided_impedance(imp)
        
    if mode == "Hermite":
        def h(f):
            return spectral_density(frequency=f, sigma=sigma, m=m,
                                    mode="Hermite")
    else:
        raise NotImplementedError("Not implemanted yet.")
    
    pmax = fmax/(ring.f0 * M) - 1
    pmin = fmin/(ring.f0 * M) + 1
    
    p = np.arange(pmin,pmax+1)
    
    if imp.component_type == "long":
        fp = ring.f0*(p*M + mu + m*tuneS)
        fp = fp[np.nonzero(fp)] # Avoid division by 0
        num = np.sum( imp(fp) * h(fp) / (fp*2*np.pi) )
        den = np.sum( h(fp) )
        Zeff = num/den
        
    elif imp.component_type == "xdip" or imp.component_type == "ydip":
        if imp.component_type == "xdip":
            tuneXY = ring.tune[0]
            if xi is None :
                xi = ring.chro[0]
        elif imp.component_type == "ydip":
            tuneXY = ring.tune[1]
            if xi is None:
                xi = ring.chro[1]
        fp = ring.f0*(p*M + mu + tuneXY + m*tuneS)
        f_xi = xi/ring.eta*ring.f0
        num = np.sum( imp(fp) * h(fp - f_xi) )
        den = np.sum( h(fp - f_xi) )
        Zeff = num/den
    else:
        raise TypeError("Effective impedance is only defined for long, xdip"
                        " and ydip impedance type.")
        
    return Zeff


def yokoya_elliptic(x_radius , y_radius):
    """
    Compute Yokoya factors for an elliptic beam pipe.
    Function adapted from N. Mounet IW2D.

    Parameters
    ----------
    x_radius : float
        Horizontal semi-axis of the ellipse in [m].
    y_radius : float
        Vertical semi-axis of the ellipse in [m].

    Returns
    -------
    yoklong : float
        Yokoya factor for the longitudinal impedance.
    yokxdip : float
        Yokoya factor for the dipolar horizontal impedance.
    yokydip : float
        Yokoya factor for the dipolar vertical impedance.
    yokxquad : float
        Yokoya factor for the quadrupolar horizontal impedance.
    yokyquad : float
        Yokoya factor for the quadrupolar vertical impedance.
    """
    if y_radius < x_radius:
        small_semiaxis = y_radius
        large_semiaxis = x_radius
    else:
        small_semiaxis = x_radius
        large_semiaxis = y_radius
        
    path_to_file = Path(__file__).parent
    file = path_to_file / "data" / "Yokoya_elliptic_from_Elias_USPAS.csv"

    # read Yokoya factors interpolation file
    # BEWARE: columns are ratio, dipy, dipx, quady, quadx
    yokoya_file = pd.read_csv(file)
    ratio_col = yokoya_file["x"]
    # compute semi-axes ratio (first column of this file)
    ratio = (large_semiaxis - small_semiaxis)/(large_semiaxis + small_semiaxis)

    # interpolate Yokoya file at the correct ratio
    yoklong = 1
    
    if y_radius < x_radius:
        yokydip = np.interp(ratio, ratio_col, yokoya_file["dipy"])
        yokxdip = np.interp(ratio, ratio_col, yokoya_file["dipx"])
        yokyquad = np.interp(ratio, ratio_col, yokoya_file["quady"])
        yokxquad = np.interp(ratio, ratio_col, yokoya_file["quadx"])
    else:
        yokxdip = np.interp(ratio, ratio_col, yokoya_file["dipy"])
        yokydip = np.interp(ratio, ratio_col, yokoya_file["dipx"])
        yokxquad = np.interp(ratio, ratio_col, yokoya_file["quady"])
        yokyquad = np.interp(ratio, ratio_col, yokoya_file["quadx"])        

    return (yoklong, yokxdip, yokydip, yokxquad, yokyquad)

def beam_loss_factor(impedance, frequency, spectrum, ring):
    """
    Compute "beam" loss factor using the beam spectrum, uses a sum instead of 
    integral compared to loss_factor [1].

    Parameters
    ----------
    impedance : Impedance of type "long"
    frequency : array
        Sample points of spectrum.
    spectrum : array
        Beam spectrum to consider.
    ring : Synchrotron object

    Returns
    -------
    kloss_beam : float
        Beam loss factor in [V/C].
        
    References
    ----------
    [1] : Handbook of accelerator physics and engineering, 3rd printing. 
        Eq (3) p239.
    """
    pmax = np.floor(impedance.data.index.max()/ring.f0)
    pmin = np.floor(impedance.data.index.min()/ring.f0)
    
    if pmin >= 0:
        double_sided_impedance(impedance)
        pmin = -1*pmax
    
    p = np.arange(pmin+1,pmax)    
    pf0 = p*ring.f0
    ReZ = np.real(impedance(pf0))
    spectral_density = np.abs(spectrum)**2
    # interpolation of the spectrum is needed to avoid problems liked to 
    # division by 0
    # computing the spectrum directly to the frequency points gives
    # wrong results
    spect = interp1d(frequency, spectral_density)
    kloss_beam = ring.f0 * np.sum(ReZ*spect(pf0))
    
    return kloss_beam

def double_sided_impedance(impedance):
    """
    Add negative frequency points to single sided impedance spectrum following
    symetries depending on impedance type.

    Parameters
    ----------
    impedance : Impedance object
        Single sided impedance.
    """
    fmin = impedance.data.index.min()
    
    if fmin >= 0:
        negative_index = impedance.data.index*-1
        negative_data = impedance.data.set_index(negative_index)
        
        imp_type = impedance.component_type
        
        if imp_type == "long":
            negative_data["imag"] = -1*negative_data["imag"]
            
        elif (imp_type == "xdip") or (imp_type == "ydip"):
            negative_data["real"] = -1*negative_data["real"]
        
        elif (imp_type == "xquad") or (imp_type == "yquad"):
            negative_data["real"] = -1*negative_data["real"]
            
        else:
            raise ValueError("Wrong impedance type")
            
        try:
            negative_data = negative_data.drop(0)
        except KeyError:
            pass
            
        all_data = impedance.data.append(negative_data)
        all_data = all_data.sort_index()
        impedance.data = all_data
        
