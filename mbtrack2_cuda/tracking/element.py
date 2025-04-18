#-*- coding: utf-8 -*-
"""
This module defines the most basic elements for tracking, including Element,
an abstract base class which is to be used as mother class to every elements
included in the tracking.
"""
import numpy as np
import numba
import os
import h5py as hp
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, \
     xoroshiro128p_normal_float32, xoroshiro128p_normal_float64
from math import ceil, sqrt, sin, cos, inf, exp
from scipy.constants import mu_0, c, pi
from abc import ABCMeta, abstractmethod
from functools import wraps
from copy import deepcopy
from mbtrack2_cuda.tracking.particles import Beam
from mbtrack2_cuda.utilities import yokoya_elliptic

class Element(metaclass=ABCMeta):
    """
    Abstract Element class used for subclass inheritance to define all kinds 
    of objects which intervene in the tracking.
    """

    @abstractmethod
    def track(self, beam):
        """
        Track a beam object through this Element.
        This method needs to be overloaded in each Element subclass.
        
        Parameters
        ----------
        beam : Beam object
        """
        raise NotImplementedError
        
    @staticmethod
    def parallel(track):
        """
        Defines the decorator @parallel which handle the embarrassingly 
        parallel case which happens when there is no bunch to bunch 
        interaction in the tracking routine.
        
        Adding @Element.parallel allows to write the track method of the 
        Element subclass for a Bunch object instead of a Beam object.
        
        Parameters
        ----------
        track : function, method of an Element subclass
            track method of an Element subclass which takes a Bunch object as
            input
            
        Returns
        -------
        track_wrapper: function, method of an Element subclass
            track method of an Element subclass which takes a Beam object or a
            Bunch object as input
        """
        @wraps(track)
        def track_wrapper(*args, **kwargs):
            if isinstance(args[1], Beam):
                self = args[0]
                beam = args[1]
                if (beam.mpi_switch == True):
                    track(self, beam[beam.mpi.bunch_num], *args[2:], **kwargs)
                elif (beam.cuda_switch == True):
                    track(self, beam, *args[2:], **kwargs)
                else:
                    for bunch in beam.not_empty:
                        track(self, bunch, *args[2:], **kwargs)
            else:
                self = args[0]
                bunch = args[1]
                track(self, bunch, *args[2:], **kwargs)
        return track_wrapper

class LongitudinalMap(Element):
    """
    Longitudinal map for a single turn in the synchrotron.
    
    Parameters
    ----------
    ring : Synchrotron object
    """
    
    def __init__(self, ring):
        self.ring = ring
        
    @Element.parallel
    def track(self, bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """
        bunch["delta"] -= self.ring.U0 / self.ring.E0
        bunch["tau"] += self.ring.ac * self.ring.T0 * bunch["delta"]

class SynchrotronRadiation(Element):
    """
    Element to handle synchrotron radiation, radiation damping and quantum 
    excitation, for a single turn in the synchrotron.
    
    Parameters
    ----------
    ring : Synchrotron object
    switch : bool array of shape (3,), optional
        allow to choose on which plane the synchrotron radiation is active
    """
    
    def __init__(self, ring, switch = np.ones((3,), dtype=bool)):
        self.ring = ring
        self.switch = switch
        
    @Element.parallel        
    def track(self, bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """
        if (self.switch[0] == True):
            rand = np.random.normal(size=len(bunch))
            bunch["delta"] = ((1 - 2*self.ring.T0/self.ring.tau[2])*bunch["delta"] +
                 2*self.ring.sigma_delta*np.sqrt(self.ring.T0/self.ring.tau[2])*rand)
            
        if (self.switch[1] == True):
            rand = np.random.normal(size=len(bunch))
            bunch["xp"] = ((1 - 2*self.ring.T0/self.ring.tau[0])*bunch["xp"] +
                 2*self.ring.sigma()[1]*np.sqrt(self.ring.T0/self.ring.tau[0])*rand)
       
        if (self.switch[2] == True):
            rand = np.random.normal(size=len(bunch))
            bunch["yp"] = ((1 - 2*self.ring.T0/self.ring.tau[1])*bunch["yp"] +
                 2*self.ring.sigma()[3]*np.sqrt(self.ring.T0/self.ring.tau[1])*rand)
        
class TransverseMap(Element):
    """
    Transverse map for a single turn in the synchrotron.
    
    Parameters
    ----------
    ring : Synchrotron object
    """
    
    def __init__(self, ring):
        self.ring = ring
        self.alpha = self.ring.optics.local_alpha
        self.beta = self.ring.optics.local_beta
        self.gamma = self.ring.optics.local_gamma
        self.dispersion = self.ring.optics.local_dispersion        
        if self.ring.adts is not None:
            self.adts_poly = [np.poly1d(self.ring.adts[0]),
                              np.poly1d(self.ring.adts[1]),
                              np.poly1d(self.ring.adts[2]), 
                              np.poly1d(self.ring.adts[3])]
    
    @Element.parallel    
    def track(self, bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """

        # Compute phase advance which depends on energy via chromaticity and ADTS
        if self.ring.adts is None:
            phase_advance_x = 2*pi * (self.ring.tune[0] + 
                                         self.ring.chro[0]*bunch["delta"])
            phase_advance_y = 2*pi * (self.ring.tune[1] + 
                                         self.ring.chro[1]*bunch["delta"])
        else:
            Jx = (self.ring.optics.local_gamma[0] * bunch['x']**2) + \
                  (2*self.ring.optics.local_alpha[0] * bunch['x']*bunch['xp']) + \
                  (self.ring.optics.local_beta[0] * bunch['xp']**2)
            Jy = (self.ring.optics.local_gamma[1] * bunch['y']**2) + \
                  (2*self.ring.optics.local_alpha[1] * bunch['y']*bunch['yp']) + \
                  (self.ring.optics.local_beta[1] * bunch['yp']**2)
            phase_advance_x = 2*pi * (self.ring.tune[0] + 
                                         self.ring.chro[0]*bunch["delta"] + 
                                         self.adts_poly[0](Jx) + 
                                         self.adts_poly[2](Jy))
            phase_advance_y = 2*pi * (self.ring.tune[1] + 
                                         self.ring.chro[1]*bunch["delta"] +
                                         self.adts_poly[1](Jx) + 
                                         self.adts_poly[3](Jy))

        
        # 6x6 matrix corresponding to (x, xp, delta, y, yp, delta)
        matrix = np.zeros((6,6,len(bunch)))
        
        # Horizontal
        matrix[0,0,:] = np.cos(phase_advance_x) + self.alpha[0]*np.sin(phase_advance_x)
        matrix[0,1,:] = self.beta[0]*np.sin(phase_advance_x)
        matrix[0,2,:] = self.dispersion[0]
        matrix[1,0,:] = -1*self.gamma[0]*np.sin(phase_advance_x)
        matrix[1,1,:] = np.cos(phase_advance_x) - self.alpha[0]*np.sin(phase_advance_x)
        matrix[1,2,:] = self.dispersion[1]
        matrix[2,2,:] = 1
        
        # Vertical
        matrix[3,3,:] = np.cos(phase_advance_y) + self.alpha[1]*np.sin(phase_advance_y)
        matrix[3,4,:] = self.beta[1]*np.sin(phase_advance_y)
        matrix[3,5,:] = self.dispersion[2]
        matrix[4,3,:] = -1*self.gamma[1]*np.sin(phase_advance_y)
        matrix[4,4,:] = np.cos(phase_advance_y) - self.alpha[1]*np.sin(phase_advance_y)
        matrix[4,5,:] = self.dispersion[3]
        matrix[5,5,:] = 1
        
        x = matrix[0,0,:]*bunch["x"] + matrix[0,1,:]*bunch["xp"] + matrix[0,2,:]*bunch["delta"]
        xp = matrix[1,0,:]*bunch["x"] + matrix[1,1,:]*bunch["xp"] + matrix[1,2,:]*bunch["delta"]
        y =  matrix[3,3,:]*bunch["y"] + matrix[3,4,:]*bunch["yp"] + matrix[3,5,:]*bunch["delta"]
        yp = matrix[4,3,:]*bunch["y"] + matrix[4,4,:]*bunch["yp"] + matrix[4,5,:]*bunch["delta"]
        
        bunch["x"] = x
        bunch["xp"] = xp
        bunch["y"] = y
        bunch["yp"] = yp
        
class SkewQuadrupole:
    """
    Thin skew quadrupole element used to introduce betatron coupling (the 
    length of the quadrupole is neglected).
    
    Parameters
    ----------
    strength : float
        Integrated strength of the skew quadrupole [m].
        
    """
    def __init__(self, strength):
        self.strength = strength
        
    @Element.parallel        
    def track(self, bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """
        
        bunch['xp'] = bunch['xp'] - self.strength * bunch['y']
        bunch['yp'] = bunch['yp'] - self.strength * bunch['x']

class TransverseMapSector(Element):
    """
    Transverse map for a sector of the synchrotron, from an initial 
    position s0 to a final position s1.

    Parameters
    ----------
    ring : Synchrotron object
        Ring parameters.
    alpha0 : array of shape (2,)
        Alpha Twiss function at the initial location of the sector.
    beta0 : array of shape (2,)
        Beta Twiss function at the initial location of the sector.
    dispersion0 : array of shape (4,)
        Dispersion function at the initial location of the sector.
    alpha1 : array of shape (2,)
        Alpha Twiss function at the final location of the sector.
    beta1 : array of shape (2,)
        Beta Twiss function at the final location of the sector.
    dispersion1 : array of shape (4,)
        Dispersion function at the final location of the sector.
    phase_diff : array of shape (2,)
        Phase difference between the initial and final location of the 
        sector.
    chro_diff : array of shape (2,)
        Chromaticity difference between the initial and final location of 
        the sector.
    adts : array of shape (4,), optional
        Amplitude-dependent tune shift of the sector, see Synchrotron class 
        for details. The default is None.

    """
    def __init__(self, ring, alpha0, beta0, dispersion0, alpha1, beta1, 
                 dispersion1, phase_diff, chro_diff, adts=None):
        self.ring = ring
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.gamma0 = (1 + self.alpha0**2)/self.beta0
        self.dispersion0 = dispersion0
        self.alpha1 = alpha1
        self.beta1 = beta1
        self.gamma1 = (1 + self.alpha1**2)/self.beta1
        self.dispersion1 = dispersion1  
        self.tune_diff = phase_diff / (2*pi)
        self.chro_diff = chro_diff
        if adts is not None:
            self.adts_poly = [np.poly1d(adts[0]),
                              np.poly1d(adts[1]),
                              np.poly1d(adts[2]), 
                              np.poly1d(adts[3])]
        else:
            self.adts_poly = None
    
    @Element.parallel    
    def track(self, bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """

        # Compute phase advance which depends on energy via chromaticity and ADTS
        if self.adts_poly is None:
            phase_advance_x = 2*pi * (self.tune_diff[0] + 
                                         self.chro_diff[0]*bunch["delta"])
            phase_advance_y = 2*pi * (self.tune_diff[1] + 
                                         self.chro_diff[1]*bunch["delta"])
        else:
            Jx = (self.gamma0[0] * bunch['x']**2) + \
                  (2*self.alpha0[0] * bunch['x']*self['xp']) + \
                  (self.beta0[0] * bunch['xp']**2)
            Jy = (self.gamma0[1] * bunch['y']**2) + \
                  (2*self.alpha0[1] * bunch['y']*bunch['yp']) + \
                  (self.beta0[1] * bunch['yp']**2)
            phase_advance_x = 2*pi * (self.tune_diff[0] + 
                                         self.chro_diff[0]*bunch["delta"] + 
                                         self.adts_poly[0](Jx) + 
                                         self.adts_poly[2](Jy))
            phase_advance_y = 2*pi * (self.tune_diff[1] + 
                                         self.chro_diff[1]*bunch["delta"] +
                                         self.adts_poly[1](Jx) + 
                                         self.adts_poly[3](Jy))
        
        # 6x6 matrix corresponding to (x, xp, delta, y, yp, delta)
        matrix = np.zeros((6,6,len(bunch)))
        
        # Horizontal
        matrix[0,0,:] = np.sqrt(self.beta1[0]/self.beta0[0])*(np.cos(phase_advance_x) + self.alpha0[0]*np.sin(phase_advance_x))
        matrix[0,1,:] = np.sqrt(self.beta0[0]*self.beta1[0])*np.sin(phase_advance_x)
        matrix[0,2,:] = self.dispersion1[0] - matrix[0,0,:]*self.dispersion0[0] - matrix[0,1,:]*self.dispersion0[1]
        matrix[1,0,:] = ((self.alpha0[0] - self.alpha1[0])*np.cos(phase_advance_x) - (1 + self.alpha0[0]*self.alpha1[0])*np.sin(phase_advance_x))/np.sqrt(self.beta0[0]*self.beta1[0])
        matrix[1,1,:] = np.sqrt(self.beta0[0]/self.beta1[0])*(np.cos(phase_advance_x) - self.alpha1[0]*np.sin(phase_advance_x))
        matrix[1,2,:] = self.dispersion1[1] - matrix[1,0,:]*self.dispersion0[0] - matrix[1,1,:]*self.dispersion0[1]
        matrix[2,2,:] = 1
        
        # Vertical
        matrix[3,3,:] = np.sqrt(self.beta1[1]/self.beta0[1])*(np.cos(phase_advance_y) + self.alpha0[1]*np.sin(phase_advance_y))
        matrix[3,4,:] = np.sqrt(self.beta0[1]*self.beta1[1])*np.sin(phase_advance_y)
        matrix[3,5,:] = self.dispersion1[2] - matrix[3,3,:]*self.dispersion0[2] - matrix[3,4,:]*self.dispersion0[3]
        matrix[4,3,:] = ((self.alpha0[1] - self.alpha1[1])*np.cos(phase_advance_y) - (1 + self.alpha0[1]*self.alpha1[1])*np.sin(phase_advance_y))/np.sqrt(self.beta0[1]*self.beta1[1])
        matrix[4,4,:] = np.sqrt(self.beta0[1]/self.beta1[1])*(np.cos(phase_advance_y) - self.alpha1[1]*np.sin(phase_advance_y))
        matrix[4,5,:] = self.dispersion1[3] - matrix[4,3,:]*self.dispersion0[2] - matrix[4,4,:]*self.dispersion0[3]
        matrix[5,5,:] = 1
        
        x = matrix[0,0,:]*bunch["x"] + matrix[0,1,:]*bunch["xp"] + matrix[0,2,:]*bunch["delta"]
        xp = matrix[1,0,:]*bunch["x"] + matrix[1,1,:]*bunch["xp"] + matrix[1,2,:]*bunch["delta"]
        y =  matrix[3,3,:]*bunch["y"] + matrix[3,4,:]*bunch["yp"] + matrix[3,5,:]*bunch["delta"]
        yp = matrix[4,3,:]*bunch["y"] + matrix[4,4,:]*bunch["yp"] + matrix[4,5,:]*bunch["delta"]
        
        bunch["x"] = x
        bunch["xp"] = xp
        bunch["y"] = y
        bunch["yp"] = yp
        
def transverse_map_sector_generator(ring, positions):
    """
    Convenience function which generate a list of TransverseMapSector elements
    from a ring with an AT lattice.
    
    Tracking through all the sectors is equivalent to a full turn (and thus to 
    the TransverseMap object).

    Parameters
    ----------
    ring : Synchrotron object
        Ring parameters, must .
    positions : array
        List of longitudinal positions in [m] to use as starting and end points
        of the TransverseMapSector elements.
        The array should contain the initial position (s=0) but not the end 
        position (s=ring.L), so like position = np.array([0, pos1, pos2, ...]).

    Returns
    -------
    sectors : list
        List of TransverseMapSector elements.

    """
    import at
    def _compute_chro(ring, pos, dp=1e-4):
        lat = deepcopy(ring.optics.lattice)
        lat.append(at.Marker("END"))
        N=len(lat)
        refpts=np.arange(N)
        *elem_neg_dp, = at.linopt2(lat, refpts=refpts, dp=-dp)
        *elem_pos_dp, = at.linopt2(lat, refpts=refpts, dp=dp)

        s = elem_neg_dp[2]["s_pos"]
        mux0 = elem_neg_dp[2]['mu'][:,0]
        mux1 = elem_pos_dp[2]['mu'][:,0]
        muy0 = elem_neg_dp[2]['mu'][:,1]
        muy1 = elem_pos_dp[2]['mu'][:,1]

        Chrox=(mux1-mux0)/(2*dp)/2/pi
        Chroy=(muy1-muy0)/(2*dp)/2/pi
        chrox = np.interp(pos, s, Chrox)
        chroy = np.interp(pos, s, Chroy)
        
        return np.array([chrox, chroy])
    
    if ring.optics.use_local_values:
        raise ValueError("The Synchrotron object must be loaded from an AT lattice")
    
    N_sec = len(positions)
    sectors = []
    for i in range(N_sec):
        alpha0 = ring.optics.alpha(positions[i])
        beta0 = ring.optics.beta(positions[i])
        dispersion0 = ring.optics.dispersion(positions[i])
        mu0 = ring.optics.mu(positions[i])
        chro0 = _compute_chro(ring, positions[i])
        if i != (N_sec - 1):
            alpha1 = ring.optics.alpha(positions[i+1])
            beta1 = ring.optics.beta(positions[i+1])
            dispersion1 = ring.optics.dispersion(positions[i+1])
            mu1 = ring.optics.mu(positions[i+1])
            chro1 = _compute_chro(ring, positions[i+1])
        else:
            alpha1 = ring.optics.alpha(positions[0])
            beta1 = ring.optics.beta(positions[0])
            dispersion1 = ring.optics.dispersion(positions[0])
            mu1 = ring.optics.mu(ring.L)
            chro1 = _compute_chro(ring, ring.L)
        phase_diff = mu1 - mu0
        chro_diff = chro1 - chro0
        sectors.append(TransverseMapSector(ring, alpha0, beta0, dispersion0, 
                     alpha1, beta1, dispersion1, phase_diff, chro_diff))
    return sectors

class CUDAMap(Element):
    """
    Longitudinal Map, Transverse Map, Synchrotron Radiation, RF Cavity, Resistive Wall for GPU calculations

    """
    def __init__(self, ring, filling_pattern, Vc1=1e6, Vc2=1e6, m1=1, m2=3, theta1=0, theta2=0, num_bin=80, num_bin_interp=6000,
                 norm_beta=1, rho=1.68e-8, radius_x=10e-3, radius_y=10e-3, length=1, wake_function_time=0, wake_function_integ_wl=0,
                 wake_function_integ_wtx=0, wake_function_integ_wty=0, t_crit=1e-12, idxs=0, r_lrrw=0,
                 x3_lrrw=0, y3_lrrw=0, x_aperture_radius=10, y_aperture_radius=10, thread_per_block=16):
        self.ring = ring
        self.filling_pattern = filling_pattern
        self.Vc1 = Vc1
        self.Vc2 = Vc2
        self.m1 = m1
        self.m2 = m2
        self.theta1 = theta1
        self.theta2 = theta2
        self.alpha = self.ring.optics.local_alpha
        self.beta = self.ring.optics.local_beta
        self.gamma = self.ring.optics.local_gamma
        self.dispersion = self.ring.optics.local_dispersion
        self.num_bin = num_bin
        self.num_bin_interp = num_bin_interp
        self.norm_beta = norm_beta
        self.rho = rho
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.length = length
        self.wake_function_time = wake_function_time
        self.wake_function_integ_wl = wake_function_integ_wl
        self.wake_function_integ_wtx = wake_function_integ_wtx
        self.wake_function_integ_wty = wake_function_integ_wty
        self.t_crit = t_crit
        self.idxs = idxs
        self.r_lrrw = r_lrrw
        self.x3_lrrw = x3_lrrw
        self.y3_lrrw = y3_lrrw
        self.x_aperture_radius = x_aperture_radius
        self.y_aperture_radius = y_aperture_radius
        self.thread_per_block = thread_per_block
        if self.ring.adts is not None:
            self.adts_poly = [np.poly1d(self.ring.adts[0]),
                              np.poly1d(self.ring.adts[1]),
                              np.poly1d(self.ring.adts[2]),
                              np.poly1d(self.ring.adts[3])]
            
    def track(self, bunch, turns, turns_lrrw, curm_ti, culm, cusr, cutm, curfmc, curfhc, cusbwake, culrrw,
              cuelliptic, curm, cugeneralwake, cugeneralwakeshortlong, curwcircularseries,
              rectangularaperture, monitordip, dblPrec6D, BunchToX, BunchToJ):
        """
        Tracking method for the element

        """
        # No Selection Error
        if (culm == False) and (cutm == False) and (cusr == False) and (curfmc == False):
            raise ValueError("There is nothing to track.")
        
        # if (cugeneralwake == True) and (cugeneralwakeshortlong == True):
        #     raise ValueError("You cannot perform cugeneralwake and cugeneralwakeshortlong simultaneously.")

        @cuda.jit(device=True, inline=True)
        def cuwofz(in_real, in_imag):
            """
            This function calculates the double precision scaled complex complementary
            error function (Faddeeva function) based on the algorithm of the FORTRAN
            function written at CERN by K. Koelbig, Program C335, 1970.

            This function returns only the real component of the result.

            Source
            https://github.com/PyCOMPLETE/PyHEADTAIL/blob/master/PyHEADTAIL/gpu/wofz.cu
            
            References
            [1] : Oeftiger, A., de Maria, R., Deniau, L., Li, K., McIntosh, E., Moneta, L.,
            Hegglin, S., Aviral, A. "Review of CPU and GPU Faddeeva Implementations." in
            Proceedings of IPAC2016, Busan, Korea (2016).
            [2] : M. Bassetti and G.A. Erskine, "Closed expression for the electric field
            of a two-dimensional Gaussian charge density." CERN-ISR-TH/80-06.

            """
            errf_const = 1.12837916709551
            xLim = 5.33
            yLim = 4.29

            x = abs(in_real)
            y = abs(in_imag)
    
            Rx = cuda.local.array(33, dtype=numba.float64)
            Ry = cuda.local.array(33, dtype=numba.float64)

            if y < yLim and x < xLim:
                q = (1.0 - y / yLim) * sqrt(1.0 - (x / xLim)**2)
                h  = 1.0 / (3.2 * q)
                nc = 7 + int(23.0 * q)
                xl = h**(1 - nc)
                xh = y + 0.5 / h
                yh = x
                nu = 10 + int(21.0 * q)
                Rx[nu] = 0.
                Ry[nu] = 0.
                for n in range(nu, 0, -1):
                    Tx = xh + n * Rx[n]
                    Ty = yh - n * Ry[n]
                    Tn = Tx**2 + Ty**2
                    Rx[n-1] = 0.5 * Tx / Tn
                    Ry[n-1] = 0.5 * Ty / Tn

                Sx = 0.
                Sy = 0.
                for n in range(nc, 0, -1):
                    Saux = Sx + xl
                    Sx = Rx[n-1] * Saux - Ry[n-1] * Sy
                    Sy = Rx[n-1] * Sy + Ry[n-1] * Saux
                    xl = h * xl

                wofz_x = errf_const * Sx
                wofz_y = errf_const * Sy

            else:
                xh = y
                yh = x
                Rx[0] = 0.
                Ry[0] = 0.
                for n in range(9, 0, -1):
                    Tx = xh + n * Rx[0]
                    Ty = yh - n * Ry[0]
                    Tn = Tx**2 + Ty**2
                    Rx[0] = 0.5 * Tx / Tn
                    Ry[0] = 0.5 * Ty / Tn

                wofz_x = errf_const * Rx[0]
                wofz_y = errf_const * Ry[0]

            if y == 0.:
                wofz_x = exp(-x**2)
            if in_imag < 0.:
                wofz_x = 2.0 * exp(y**2 - x**2) * cos(2.0 * x * y) - wofz_x
                wofz_y = -2.0 * exp(y**2 - x**2) * sin(2.0 * x * y) - wofz_y
                if in_real > 0.:
                    wofz_y = -wofz_y
            elif in_real < 0.:
                wofz_y = -wofz_y

            return wofz_x, wofz_y
        
        @cuda.jit(device=True, inline=True)
        def wl_integ(amp_wl_integ, pi, norm_t):
            """
            Integrated all-range longitudinal wake function

            """
            w1_real, _ = cuwofz(0, sqrt(2*norm_t))
            w2_real, w2_imag = cuwofz(cos(pi/6)*sqrt(2*norm_t), sin(pi/6)*sqrt(2*norm_t))

            res = amp_wl_integ * ( exp(-norm_t)*( -cos(sqrt(3)*norm_t)
                  + sqrt(3)*sin(sqrt(3)*norm_t) )
                  + 0.5*w1_real + cos(-pi/3)*w2_real - sin(-pi/3)*w2_imag )
            
            return res
        
        @cuda.jit(device=True, inline=True)
        def wl_25_integ(amp_common, amp_wl_25_integ, norm_t):
            """
            Integrated series expanded short-range longitudinal wake function up to 25th order

            """
            return ( amp_wl_25_integ * ( norm_t + (0.7598356856515925*norm_t)**4 + (0.5359287586710165*norm_t)**7
                    + (0.4120507346730906*norm_t)**10 + (0.33455510713488373*norm_t)**13 + (0.28162869082531644*norm_t)**16
                    + (0.24320405138468126*norm_t)**19 + (0.2140394306716979*norm_t)**22 + (0.19114501167286455*norm_t)**25
                    + (0.17269266635870276*norm_t)**28 + (0.1575021151157068*norm_t)**31 + (0.1447773898967433*norm_t)**34
                    + (0.13396219114895203*norm_t)**37 + (0.12465598050014402*norm_t)**40 + (0.11656301461446554*norm_t)**43
                    + (0.10946016271439554*norm_t)**46 + (0.10317589948854768*norm_t)**49 + (0.09757619367433015*norm_t)**52
                    + (0.09255478832390489*norm_t)**55 + (0.08802635663195192*norm_t)**58 + (0.08392158727829205*norm_t)**61
                    + (0.08018359300364696*norm_t)**64 + (0.0767652445236992*norm_t)**67 + (0.07362716301183045*norm_t)**70
                    + (0.07073618881295525*norm_t)**73 + (0.06806419956149919*norm_t)**76
                    - amp_common * ( (1.3540150378633087*norm_t)**(2.5) + (0.7442661464544931*norm_t)**(5.5)
                    + (0.5191799001499797*norm_t)**(8.5) + (0.39999929856410843*norm_t)**(11.5) + (0.3258190751474901*norm_t)**(14.5)
                    + (0.2750750366561003*norm_t)**(17.5) + (0.23812561705843846*norm_t)**(20.5) + (0.2099953322778044*norm_t)**(23.5)
                    + (0.1878511040161531*norm_t)**(26.5) + (0.16995902532650953*norm_t)**(29.5) + (0.1551974739995203*norm_t)**(32.5)
                    + (0.14280831253919998*norm_t)**(35.5) + (0.13226044275007684*norm_t)**(38.5) + (0.12317060570899128*norm_t)**(41.5)
                    + (0.11525521171024858*norm_t)**(44.5) + (0.10829987310065792*norm_t)**(47.5) + (0.10213948113200559*norm_t)**(50.5)
                    + (0.0966447981614343*norm_t)**(53.5) + (0.09171320586393214*norm_t)**(56.5) + (0.0872621791575225*norm_t)**(59.5)
                    + (0.08322459210232791*norm_t)**(62.5) + (0.07954528217155166*norm_t)**(65.5) + (0.07617849588383406*norm_t)**(68.5)
                    + (0.07308596265317083*norm_t)**(71.5) + (0.07023542357943872*norm_t)**(74.5) ) ) )
        
        @cuda.jit(device=True, inline=True)
        def wl_long_integ(amp_wl_long_integ, t):
            """
            Integrated long-range longitudinal wake function

            """

            return ( amp_wl_long_integ / sqrt(t) )
        
        @cuda.jit(device=True, inline=True)
        def wt_integ(amp_wt_integ, pi, norm_t):
            """
            Integrated all-range transverse wake function

            """
            w1_real, _ = cuwofz(0, sqrt(2*norm_t))
            w2_real, w2_imag = cuwofz(cos(pi/6)*sqrt(2*norm_t), sin(pi/6)*sqrt(2*norm_t))

            res = amp_wt_integ * ( -exp(-norm_t)*( cos(sqrt(3)*norm_t)
                  + sqrt(3)*sin(sqrt(3)*norm_t) ) + 3*sqrt(2*norm_t/pi)
                  + 0.5*w1_real - cos(-2*pi/3)*w2_real + sin(-2*pi/3)*w2_imag )
            
            return res
        
        @cuda.jit(device=True, inline=True)
        def wlwt_integ(amp_wl_integ, amp_wt_integ, pi, norm_t):
            """
            Integrated all-range longitudinal and transverse wake functions
            """

            w1_real, _ = cuwofz(0, sqrt(2*norm_t))
            w2_real, w2_imag = cuwofz(cos(pi/6)*sqrt(2*norm_t), sin(pi/6)*sqrt(2*norm_t))

            wl = amp_wl_integ * ( exp(-norm_t)*( -cos(sqrt(3)*norm_t)
                  + sqrt(3)*sin(sqrt(3)*norm_t) )
                  + 0.5*w1_real + cos(-pi/3)*w2_real - sin(-pi/3)*w2_imag )

            wt = amp_wt_integ * ( -exp(-norm_t)*( cos(sqrt(3)*norm_t)
                  + sqrt(3)*sin(sqrt(3)*norm_t) ) + 3*sqrt(2*norm_t/pi)
                  + 0.5*w1_real - cos(-2*pi/3)*w2_real + sin(-2*pi/3)*w2_imag )

            return wl, wt
        
        @cuda.jit(device=True, inline=True)
        def wt_24_integ(amp_common, amp_wt_24_integ, norm_t):
            """
            Integrated series expanded short-range transeverse wake function up to 24th order

            """
            return ( amp_wt_24_integ * ( (norm_t)**2 + (0.6683250619582689*norm_t)**5 + (0.4872043764072933*norm_t)**8
                    + (0.3825158859928969*norm_t)**11 + (0.3148256536179411*norm_t)**14 + (0.2675328660272652*norm_t)**17
                    + (0.23263350551030879*norm_t)**20 + (0.20581899606404036*norm_t)**23 + (0.1845690510871302*norm_t)**26
                    + (0.1673122049640277*norm_t)**29 + (0.15301797178571788*norm_t)**32 + (0.1409825458302609*norm_t)**35
                    + (0.13070884691318696*norm_t)**38 + (0.12183579459303688*norm_t)**41 + (0.11409475823089159*norm_t)**44
                    + (0.10728173600722947*norm_t)**47 + (0.1012390120257816*norm_t)**50 + (0.09584272666342507*norm_t)**53
                    + (0.09099424938745514*norm_t)**56 + (0.08661406256336764*norm_t)**59 + (0.08263734289994137*norm_t)**62
                    + (0.07901071498221277*norm_t)**65 + (0.07568982942981432*norm_t)**68 + (0.07263753117133885*norm_t)**71
                    + (0.0698224565799788*norm_t)**74
                    - amp_common * ( (1.0582233361536897*norm_t)**(3.5) + (0.6496973448961636*norm_t)**(6.5)
                    + (0.47212120209670166*norm_t)**(9.5) + (0.3717269856378805*norm_t)**(12.5) + (0.30691850475295435*norm_t)**(15.5)
                    + (0.26153344728733774*norm_t)**(18.5) + (0.2279393962651072*norm_t)**(21.5) + (0.20205092180662015*norm_t)**(24.5)
                    + (0.18147955223738668*norm_t)**(27.5) + (0.16473395095951127*norm_t)**(30.5) + (0.15083411861745583*norm_t)**(33.5)
                    + (0.13910917629947991*norm_t)**(36.5) + (0.12908419903002075*norm_t)**(39.5) + (0.12041343226064105*norm_t)**(42.5)
                    + (0.11283910954253322*norm_t)**(45.5) + (0.10616509887370504*norm_t)**(48.5) + (0.10023949566583547*norm_t)**(51.5)
                    + (0.09494280568853826*norm_t)**(54.5) + (0.09017972838243976*norm_t)**(57.5) + (0.0858733216196809*norm_t)**(60.5)
                    + (0.08196077915766899*norm_t)**(63.5) + (0.07839032331747328*norm_t)**(66.5) + (0.07511888349770383*norm_t)**(69.5)
                    + (0.07211033788347378*norm_t)**(72.5) ) ) )
        
        @cuda.jit(device=True, inline=True)
        def wt_long_integ(amp_wt_long_integ, t):
            """
            Integrated long-range transverse wake function

            """

            return ( amp_wt_long_integ * sqrt(t) )
        
        @cuda.jit(device=True, inline=True)
        def wl_long(amp_wl_long, t):
            """
            Long-range longitudinal wake function

            """

            return ( amp_wl_long / t**1.5 )
        
        @cuda.jit(device=True, inline=True)
        def wt_long(amp_wt_long, t):
            """
            Long-range Transverse wake function

            """

            return ( amp_wt_long / sqrt(t) )

        @cuda.jit
        def longmap1_kernel(Bij, num_particle, num_bunch, device_delta, U0, E0):
            """
            Longitudinal map for delta
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    cuda.atomic.sub(device_delta, (i, j), U0/E0)
            else:
                if i < num_bunch and j < num_particle:
                    cuda.atomic.sub(device_delta, (j, i), U0/E0)

        @cuda.jit
        def longmap2_kernel(Bij, num_particle, num_bunch, device_tau, device_delta, ac, T0):
            """
            Longitudinal map for tau
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    cuda.atomic.add(device_tau, (i, j), ac*T0*device_delta[i, j])
            else:
                if i < num_bunch and j < num_particle:
                    cuda.atomic.add(device_tau, (j, i), ac*T0*device_delta[j, i])
        
        @cuda.jit
        def transmap_kernel(Bij, num_particle, num_bunch, device_x, device_xp, device_y, device_yp, device_delta, device_x_tm,
                            device_xp_tm, device_y_tm, device_yp_tm, dispersion_x, dispersion_xp, dispersion_y, dispersion_yp,
                            tune_x, tune_y, chro_x, chro_y, pi, alpha_x, alpha_y, beta_x, beta_y, gamma_x, gamma_y):
            """
            Transverse map
            adts effects are ignored. (Future work)
            """
            i, j = cuda.grid(2)
            
            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    device_x_tm[i, j] = ( ( cos(2 * pi * (tune_x + chro_x * device_delta[i, j])) +
                                        alpha_x * sin(2 * pi * (tune_x + chro_x * device_delta[i, j])) )
                                        * device_x[i, j] + ( beta_x * sin(2 * pi * (tune_x + chro_x * device_delta[i, j])) )
                                        * device_xp[i, j] + dispersion_x * device_delta[i, j] )
                    device_xp_tm[i, j] = ( ( -1 * gamma_x * sin(2 * pi * (tune_x + chro_x * device_delta[i, j])) )
                                        * device_x[i, j] + ( cos(2 * pi * (tune_x + chro_x * device_delta[i, j])) -
                                        alpha_x * sin(2 * pi * (tune_x + chro_x * device_delta[i, j])) ) * device_xp[i, j] +
                                        dispersion_xp * device_delta[i, j] )
                    device_y_tm[i, j] = ( ( cos(2 * pi * (tune_y + chro_y * device_delta[i, j])) +
                                        alpha_y * sin(2 * pi * (tune_y + chro_y * device_delta[i, j])) )
                                        * device_y[i, j] + ( beta_y * sin(2 * pi * (tune_y + chro_y * device_delta[i, j])) )
                                        * device_yp[i, j] + dispersion_y * device_delta[i, j] )
                    device_yp_tm[i, j] = ( ( -1 * gamma_y * sin(2 * pi * (tune_y + chro_y * device_delta[i, j])) )
                                        * device_y[i, j] + ( cos(2 * pi * (tune_y + chro_y * device_delta[i, j])) -
                                        alpha_y * sin(2 * pi * (tune_y + chro_y * device_delta[i, j])) ) * device_yp[i, j] +
                                        dispersion_yp * device_delta[i, j] )
            else:
                if i < num_bunch and j < num_particle:
                    device_x_tm[j, i] = ( ( cos(2 * pi * (tune_x + chro_x * device_delta[j, i])) +
                                        alpha_x * sin(2 * pi * (tune_x + chro_x * device_delta[j, i])) )
                                        * device_x[j, i] + ( beta_x * sin(2 * pi * (tune_x + chro_x * device_delta[j, i])) )
                                        * device_xp[j, i] + dispersion_x * device_delta[j, i] )
                    device_xp_tm[j, i] = ( ( -1 * gamma_x * sin(2 * pi * (tune_x + chro_x * device_delta[j, i])) )
                                        * device_x[j, i] + ( cos(2 * pi * (tune_x + chro_x * device_delta[j, i])) -
                                        alpha_x * sin(2 * pi * (tune_x + chro_x * device_delta[j, i])) ) * device_xp[j, i] +
                                        dispersion_xp * device_delta[j, i] )
                    device_y_tm[j, i] = ( ( cos(2 * pi * (tune_y + chro_y * device_delta[j, i])) +
                                        alpha_y * sin(2 * pi * (tune_y + chro_y * device_delta[j, i])) )
                                        * device_y[j, i] + ( beta_y * sin(2 * pi * (tune_y + chro_y * device_delta[j, i])) )
                                        * device_yp[j, i] + dispersion_y * device_delta[j, i] )
                    device_yp_tm[j, i] = ( ( -1 * gamma_y * sin(2 * pi * (tune_y + chro_y * device_delta[j, i])) )
                                        * device_y[j, i] + ( cos(2 * pi * (tune_y + chro_y * device_delta[j, i])) -
                                        alpha_y * sin(2 * pi * (tune_y + chro_y * device_delta[j, i])) ) * device_yp[j, i] +
                                        dispersion_yp * device_delta[j, i] )
        
        @cuda.jit
        def tm_conversion_kernel(Bij, num_particle, num_bunch, device_x_tm, device_xp_tm, device_y_tm, device_yp_tm,
                                 device_x, device_xp, device_y, device_yp):
            """
            Conversion of tm arrays
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    device_x[i, j] = device_x_tm[i, j]
                    device_xp[i, j] = device_xp_tm[i, j]
                    device_y[i, j] = device_y_tm[i, j]
                    device_yp[i, j] = device_yp_tm[i, j]
            else:
                if i < num_bunch and j < num_particle:
                    device_x[j, i] = device_x_tm[j, i]
                    device_xp[j, i] = device_xp_tm[j, i]
                    device_y[j, i] = device_y_tm[j, i]
                    device_yp[j, i] = device_yp_tm[j, i]

        @cuda.jit
        def aperture_tm_conversion_kernel(Bij, num_particle, num_bunch, device_x_tm, device_xp_tm, device_y_tm, device_yp_tm,
                                 device_x, device_xp, device_y, device_yp, device_aperture):
            """
            Conversion of tm arrays
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    if device_aperture[i, j] == 1:
                        device_x[i, j] = device_x_tm[i, j]
                        device_xp[i, j] = device_xp_tm[i, j]
                        device_y[i, j] = device_y_tm[i, j]
                        device_yp[i, j] = device_yp_tm[i, j]
            else:
                if i < num_bunch and j < num_particle:
                    if device_aperture[j, i] == 1:
                        device_x[j, i] = device_x_tm[j, i]
                        device_xp[j, i] = device_xp_tm[j, i]
                        device_y[j, i] = device_y_tm[j, i]
                        device_yp[j, i] = device_yp_tm[j, i]

        @cuda.jit
        def rectangular_aperture_kernel(Bij, num_particle, num_bunch, device_x, device_y, x_aperture_radius, y_aperture_radius,
                                        device_aperture):
            """
            The particles which are outside of the rectangle are lost and not used in the tracking anymore.
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    if abs(device_x[i, j]) >= x_aperture_radius or abs(device_y[i, j]) >= y_aperture_radius:
                        device_aperture[i, j] = 0
            else:
                if i < num_bunch and j < num_particle:
                    if abs(device_x[j, i]) >= x_aperture_radius or abs(device_y[j, i]) >= y_aperture_radius:
                        device_aperture[j, i] = 0

        @cuda.jit
        def rng_kernel32(Bij, num_particle, num_bunch, device_xp, device_yp, device_delta, device_xp_sr, device_yp_sr,
                       device_delta_sr, device_rand_xp, device_rand_yp, device_rand_delta, sigma_xp, sigma_yp,
                       sigma_delta, T0, tau_h, tau_v, tau_l, turns, rng_states1, k):
            """
            Random number generation for synchrotron radiation and radiation damping
            Partial Calculations for sr to avoid the race conditions
            This approach not only avoids facing race conditions but also eliminates the need for performing
            atomic operations.
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    device_rand_xp[i] = 2*sigma_xp*sqrt(T0/tau_h)*xoroshiro128p_normal_float32(
                                        rng_states1, (i + k) )
                    device_rand_yp[i] = 2*sigma_yp*sqrt(T0/tau_v)*xoroshiro128p_normal_float32(
                                        rng_states1, (i + k + num_bunch + turns - 1) )
                    device_rand_delta[i] = 2*sigma_delta*sqrt(T0/tau_l)*xoroshiro128p_normal_float32(
                                        rng_states1, (i + k + 2*(num_bunch + turns - 1)) )

                    device_xp_sr[i, j] = (1 - 2*T0/tau_h) * device_xp[i, j]
                    device_yp_sr[i, j] = (1 - 2*T0/tau_v) * device_yp[i, j]
                    device_delta_sr[i, j] = (1 - 2*T0/tau_l) * device_delta[i, j]

            else:
                if i < num_bunch and j < num_particle:
                    device_rand_xp[j] = 2*sigma_xp*sqrt(T0/tau_h)*xoroshiro128p_normal_float32(
                                        rng_states1, (j + k) )
                    device_rand_yp[j] = 2*sigma_yp*sqrt(T0/tau_v)*xoroshiro128p_normal_float32(
                                        rng_states1, (j + k + num_bunch + turns - 1) )
                    device_rand_delta[j] = 2*sigma_delta*sqrt(T0/tau_l)*xoroshiro128p_normal_float32(
                                        rng_states1, (j + k + 2*(num_bunch + turns - 1)) )

                    device_xp_sr[j, i] = (1 - 2*T0/tau_h) * device_xp[j, i]
                    device_yp_sr[j, i] = (1 - 2*T0/tau_v) * device_yp[j, i]
                    device_delta_sr[j, i] = (1 - 2*T0/tau_l) * device_delta[j, i]
        
        @cuda.jit
        def rng_kernel64(Bij, num_particle, num_bunch, device_xp, device_yp, device_delta, device_xp_sr, device_yp_sr,
                       device_delta_sr, device_rand_xp, device_rand_yp, device_rand_delta, sigma_xp, sigma_yp,
                       sigma_delta, T0, tau_h, tau_v, tau_l, turns, rng_states1, k):
            """
            Random number generation for synchrotron radiation and radiation damping
            Partial Calculations for sr to avoid the race conditions
            This approach not only avoids facing race conditions but also eliminates the need for performing
            atomic operations.
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    device_rand_xp[i] = 2*sigma_xp*sqrt(T0/tau_h)*xoroshiro128p_normal_float64(
                                        rng_states1, (i + k) )
                    device_rand_yp[i] = 2*sigma_yp*sqrt(T0/tau_v)*xoroshiro128p_normal_float64(
                                        rng_states1, (i + k + num_bunch + turns - 1) )
                    device_rand_delta[i] = 2*sigma_delta*sqrt(T0/tau_l)*xoroshiro128p_normal_float64(
                                        rng_states1, (i + k + 2*(num_bunch + turns - 1)) )

                    device_xp_sr[i, j] = (1 - 2*T0/tau_h) * device_xp[i, j]
                    device_yp_sr[i, j] = (1 - 2*T0/tau_v) * device_yp[i, j]
                    device_delta_sr[i, j] = (1 - 2*T0/tau_l) * device_delta[i, j]

            else:
                if i < num_bunch and j < num_particle:
                    device_rand_xp[j] = 2*sigma_xp*sqrt(T0/tau_h)*xoroshiro128p_normal_float64(
                                        rng_states1, (j + k) )
                    device_rand_yp[j] = 2*sigma_yp*sqrt(T0/tau_v)*xoroshiro128p_normal_float64(
                                        rng_states1, (j + k + num_bunch + turns - 1) )
                    device_rand_delta[j] = 2*sigma_delta*sqrt(T0/tau_l)*xoroshiro128p_normal_float64(
                                        rng_states1, (j + k + 2*(num_bunch + turns - 1)) )

                    device_xp_sr[j, i] = (1 - 2*T0/tau_h) * device_xp[j, i]
                    device_yp_sr[j, i] = (1 - 2*T0/tau_v) * device_yp[j, i]
                    device_delta_sr[j, i] = (1 - 2*T0/tau_l) * device_delta[j, i]

        @cuda.jit
        def sr_kernel(Bij, num_particle, num_bunch, device_rand_xp, device_rand_yp, device_rand_delta, device_xp_sr,
                      device_yp_sr, device_delta_sr, device_xp, device_yp, device_delta):
            """
            Synchrotron radiation & radiation damping
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    device_xp[i, j] = device_xp_sr[i, j] + device_rand_xp[i]
                    device_yp[i, j] = device_yp_sr[i, j] + device_rand_yp[i]
                    device_delta[i, j] = device_delta_sr[i, j] + device_rand_delta[i]
            else:
                if i < num_bunch and j < num_particle:
                    device_xp[j, i] = device_xp_sr[j, i] + device_rand_xp[j]
                    device_yp[j, i] = device_yp_sr[j, i] + device_rand_yp[j]
                    device_delta[j, i] = device_delta_sr[j, i] + device_rand_delta[j]

        @cuda.jit
        def rfc_kernel(Bij, num_particle, num_bunch, device_tau, device_delta, omega1, E0, Vc1, Vc2, m1, m2, theta1, theta2, curfhc):
            """
            RF main cavity
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    if curfhc:
                        cuda.atomic.add(device_delta, (i, j), Vc1/E0*cos(m1*omega1*device_tau[i, j]+theta1)
                                        +Vc2/E0*cos(m2*omega1*device_tau[i, j]+theta2))
                    else:
                        cuda.atomic.add(device_delta, (i, j), Vc1/E0*cos(m1*omega1*device_tau[i, j]+theta1))
                
            else:
                if i < num_bunch and j < num_particle:
                    if curfhc:
                        cuda.atomic.add(device_delta, (j, i), Vc1/E0*cos(m1*omega1*device_tau[j, i]+theta1)
                                        +Vc2/E0*cos(m2*omega1*device_tau[j, i]+theta2))
                    else:
                        cuda.atomic.add(device_delta, (j, i), Vc1/E0*cos(m1*omega1*device_tau[j, i]+theta1))

        @cuda.jit
        def mm_pr_kernel32(num_bunch, device_tau, device_prefix_min_tau, device_prefix_max_tau, Jij):
            """
            1st parallel reduction of min & max finding for each bunch

            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y
            
            shared_min_tau = cuda.shared.array(threadperblock, numba.float32)
            shared_max_tau = cuda.shared.array(threadperblock, numba.float32)
            
            if Jij == 1:
                shared_min_tau[local_i, local_j] = device_tau[i, j]
                shared_max_tau[local_i, local_j] = device_tau[i, j]
                cuda.syncthreads()

                s = threadperblock[0]
                s >>= 1
                while s > 0:
                    if local_i < s and local_j < threadperblock[1]:
                        shared_min_tau[local_i, local_j] = min(shared_min_tau[local_i, local_j], shared_min_tau[local_i + s, local_j])
                        shared_max_tau[local_i, local_j] = max(shared_max_tau[local_i, local_j], shared_max_tau[local_i + s, local_j])
                    cuda.syncthreads()
                    s >>= 1
                
                if local_i == 0 and j < num_bunch:
                    device_prefix_min_tau[cuda.blockIdx.x, j] = shared_min_tau[0, local_j]
                    device_prefix_max_tau[cuda.blockIdx.x, j] = shared_max_tau[0, local_j]
            
            else:
                shared_min_tau[local_j, local_i] = device_tau[j, i]
                shared_max_tau[local_j, local_i] = device_tau[j, i]
                cuda.syncthreads()

                s = threadperblock[1]
                s >>= 1
                while s > 0:
                    if local_j < s and local_i < threadperblock[0]:
                        shared_min_tau[local_j, local_i] = min(shared_min_tau[local_j, local_i], shared_min_tau[local_j + s, local_i])
                        shared_max_tau[local_j, local_i] = max(shared_max_tau[local_j, local_i], shared_max_tau[local_j + s, local_i])
                    cuda.syncthreads()
                    s >>= 1
                
                if local_j == 0 and i < num_bunch:
                    device_prefix_min_tau[cuda.blockIdx.y, i] = shared_min_tau[0, local_i]
                    device_prefix_max_tau[cuda.blockIdx.y, i] = shared_max_tau[0, local_i]
        
        @cuda.jit
        def mm_pr_kernel64(num_bunch, device_tau, device_prefix_min_tau, device_prefix_max_tau, Jij):
            """
            1st parallel reduction of min & max finding for each bunch

            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y
            
            shared_min_tau = cuda.shared.array(threadperblock, numba.float64)
            shared_max_tau = cuda.shared.array(threadperblock, numba.float64)
            
            if Jij == 1:
                shared_min_tau[local_i, local_j] = device_tau[i, j]
                shared_max_tau[local_i, local_j] = device_tau[i, j]
                cuda.syncthreads()

                s = threadperblock[0]
                s >>= 1
                while s > 0:
                    if local_i < s and local_j < threadperblock[1]:
                        shared_min_tau[local_i, local_j] = min(shared_min_tau[local_i, local_j], shared_min_tau[local_i + s, local_j])
                        shared_max_tau[local_i, local_j] = max(shared_max_tau[local_i, local_j], shared_max_tau[local_i + s, local_j])
                    cuda.syncthreads()
                    s >>= 1
                
                if local_i == 0 and j < num_bunch:
                    device_prefix_min_tau[cuda.blockIdx.x, j] = shared_min_tau[0, local_j]
                    device_prefix_max_tau[cuda.blockIdx.x, j] = shared_max_tau[0, local_j]
            
            else:
                shared_min_tau[local_j, local_i] = device_tau[j, i]
                shared_max_tau[local_j, local_i] = device_tau[j, i]
                cuda.syncthreads()

                s = threadperblock[1]
                s >>= 1
                while s > 0:
                    if local_j < s and local_i < threadperblock[0]:
                        shared_min_tau[local_j, local_i] = min(shared_min_tau[local_j, local_i], shared_min_tau[local_j + s, local_i])
                        shared_max_tau[local_j, local_i] = max(shared_max_tau[local_j, local_i], shared_max_tau[local_j + s, local_i])
                    cuda.syncthreads()
                    s >>= 1
                
                if local_j == 0 and i < num_bunch:
                    device_prefix_min_tau[cuda.blockIdx.y, i] = shared_min_tau[0, local_i]
                    device_prefix_max_tau[cuda.blockIdx.y, i] = shared_max_tau[0, local_i]

        @cuda.jit
        def initialize_gm_kernel(device_prefix_min_tau, device_prefix_max_tau, device_axis_min_tau, device_axis_max_tau,
                                 device_axis_sum_x, device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                 device_axis_sum_y, device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                 device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                 device_axis_sum_delta, device_density_profile, device_profile, device_sum_bin_x,
                                 device_sum_bin_y, device_wl_avg, device_wt_avg, device_wp_x, device_wp_y,
                                 device_wp_tau, device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, device_axis_sum_tau_lrrw,
                                 device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau,
                                 Bij, num_bunch, num_bin, num_bin_interp, k):
            """
            Initialize global memory arrays

            """
            # num_particle should be larger than 2*num_bin_interp-1.
            i, j = cuda.grid(2)
            
            if Bij == 0:
                if k == 0:
                    if j < num_bunch:
                        device_axis_min_tau[j] = device_prefix_min_tau[0, j]
                        device_axis_max_tau[j] = device_prefix_max_tau[0, j]
                        device_axis_sum_x[j] = 0
                        device_axis_sum_x_squared[j] = 0
                        device_axis_sum_xp_squared[j] = 0
                        device_axis_sum_x_xp[j] = 0
                        device_axis_sum_y[j] = 0
                        device_axis_sum_y_squared[j] = 0
                        device_axis_sum_yp_squared[j] = 0
                        device_axis_sum_y_yp[j] = 0
                        device_axis_sum_tau_squared[j] = 0
                        device_axis_sum_delta_squared[j] = 0
                        device_axis_sum_tau[j] = 0
                        device_axis_sum_delta[j] = 0
                        device_axis_sum_x_lrrw[j] = 0
                        device_axis_sum_y_lrrw[j] = 0
                        device_axis_sum_tau_lrrw[j] = 0
                        device_sum_kick_x[j] = 0
                        device_sum_kick_y[j] = 0
                        device_sum_kick_tau[j] = 0
                        if i < num_bin:
                            device_profile[i, j] = 0
                            device_density_profile[i, j] = 0
                            device_sum_bin_x[i, j] = 0
                            device_sum_bin_y[i, j] = 0
                        if i < num_bin_interp:
                            device_wp_x[i, j] = 0
                            device_wp_y[i, j] = 0
                            device_wp_tau[i, j] = 0
                        if (i < 2*num_bin_interp-1):
                            device_wt_avg[i, j] = 0
                            device_wl_avg[i, j] = 0
                else:
                    if j < num_bunch:
                        device_axis_min_tau[j] = device_prefix_min_tau[0, j]
                        device_axis_max_tau[j] = device_prefix_max_tau[0, j]
                        device_axis_sum_x[j] = 0
                        device_axis_sum_x_squared[j] = 0
                        device_axis_sum_xp_squared[j] = 0
                        device_axis_sum_x_xp[j] = 0
                        device_axis_sum_y[j] = 0
                        device_axis_sum_y_squared[j] = 0
                        device_axis_sum_yp_squared[j] = 0
                        device_axis_sum_y_yp[j] = 0
                        device_axis_sum_tau_squared[j] = 0
                        device_axis_sum_delta_squared[j] = 0
                        device_axis_sum_tau[j] = 0
                        device_axis_sum_delta[j] = 0
                        device_axis_sum_x_lrrw[j] = 0
                        device_axis_sum_y_lrrw[j] = 0
                        device_axis_sum_tau_lrrw[j] = 0
                        device_sum_kick_x[j] = 0
                        device_sum_kick_y[j] = 0
                        device_sum_kick_tau[j] = 0
                        if i < num_bin:
                            device_profile[i, j] = 0
                            device_density_profile[i, j] = 0
                            device_sum_bin_x[i, j] = 0
                            device_sum_bin_y[i, j] = 0
                        if i < num_bin_interp:
                            device_wp_x[i, j] = 0
                            device_wp_y[i, j] = 0
                            device_wp_tau[i, j] = 0

            else:
                if k == 0:
                    if i < num_bunch:
                        device_axis_min_tau[i] = device_prefix_min_tau[0, i]
                        device_axis_max_tau[i] = device_prefix_max_tau[0, i]
                        device_axis_sum_x[i] = 0
                        device_axis_sum_x_squared[i] = 0
                        device_axis_sum_xp_squared[i] = 0
                        device_axis_sum_x_xp[i] = 0
                        device_axis_sum_y[i] = 0
                        device_axis_sum_y_squared[i] = 0
                        device_axis_sum_yp_squared[i] = 0
                        device_axis_sum_y_yp[i] = 0
                        device_axis_sum_tau_squared[i] = 0
                        device_axis_sum_delta_squared[i] = 0
                        device_axis_sum_tau[i] = 0
                        device_axis_sum_delta[i] = 0
                        device_axis_sum_x_lrrw[i] = 0
                        device_axis_sum_y_lrrw[i] = 0
                        device_axis_sum_tau_lrrw[i] = 0
                        device_sum_kick_x[i] = 0
                        device_sum_kick_y[i] = 0
                        device_sum_kick_tau[i] = 0
                        if j < num_bin:
                            device_profile[j, i] = 0
                            device_density_profile[j, i] = 0
                            device_sum_bin_x[j, i] = 0
                            device_sum_bin_y[j, i] = 0
                        if j < num_bin_interp:
                            device_wp_x[j, i] = 0
                            device_wp_y[j, i] = 0
                            device_wp_tau[j, i] = 0
                        if (j < 2*num_bin_interp-1):
                            device_wt_avg[j, i] = 0
                            device_wl_avg[j, i] = 0
                else:
                    if i < num_bunch:
                        device_axis_min_tau[i] = device_prefix_min_tau[0, i]
                        device_axis_max_tau[i] = device_prefix_max_tau[0, i]
                        device_axis_sum_x[i] = 0
                        device_axis_sum_x_squared[i] = 0
                        device_axis_sum_xp_squared[i] = 0
                        device_axis_sum_x_xp[i] = 0
                        device_axis_sum_y[i] = 0
                        device_axis_sum_y_squared[i] = 0
                        device_axis_sum_yp_squared[i] = 0
                        device_axis_sum_y_yp[i] = 0
                        device_axis_sum_tau_squared[i] = 0
                        device_axis_sum_delta_squared[i] = 0
                        device_axis_sum_tau[i] = 0
                        device_axis_sum_delta[i] = 0
                        device_axis_sum_x_lrrw[i] = 0
                        device_axis_sum_y_lrrw[i] = 0
                        device_axis_sum_tau_lrrw[i] = 0
                        device_sum_kick_x[i] = 0
                        device_sum_kick_y[i] = 0
                        device_sum_kick_tau[i] = 0
                        if j < num_bin:
                            device_profile[j, i] = 0
                            device_density_profile[j, i] = 0
                            device_sum_bin_x[j, i] = 0
                            device_sum_bin_y[j, i] = 0
                        if j < num_bin_interp:
                            device_wp_x[j, i] = 0
                            device_wp_y[j, i] = 0
                            device_wp_tau[j, i] = 0

        @cuda.jit
        def aperture_initialize_gm_kernel(inf, device_num_particle_alive, device_axis_min_tau, device_axis_max_tau,
                                          device_axis_sum_x, device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                          device_axis_sum_y, device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                          device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                          device_axis_sum_delta, device_density_profile, device_profile, device_sum_bin_x,
                                          device_sum_bin_y, device_wl_avg, device_wt_avg, device_wp_x, device_wp_y,
                                          device_wp_tau, device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, device_axis_sum_tau_lrrw,
                                          device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau,
                                          Bij, num_bunch, num_bin, num_bin_interp, k):
            """
            Initialize global memory arrays when the beam loss is considered

            """
            # num_particle should be larger than 2*num_bin_interp-1.
            i, j = cuda.grid(2)

            if Bij == 0:
                if k == 0:
                    if j < num_bunch:
                        device_axis_min_tau[j] = inf
                        device_axis_max_tau[j] = -inf
                        device_num_particle_alive[j] = 0
                        device_axis_sum_x[j] = 0
                        device_axis_sum_x_squared[j] = 0
                        device_axis_sum_xp_squared[j] = 0
                        device_axis_sum_x_xp[j] = 0
                        device_axis_sum_y[j] = 0
                        device_axis_sum_y_squared[j] = 0
                        device_axis_sum_yp_squared[j] = 0
                        device_axis_sum_y_yp[j] = 0
                        device_axis_sum_tau_squared[j] = 0
                        device_axis_sum_delta_squared[j] = 0
                        device_axis_sum_tau[j] = 0
                        device_axis_sum_delta[j] = 0
                        device_axis_sum_x_lrrw[j] = 0
                        device_axis_sum_y_lrrw[j] = 0
                        device_axis_sum_tau_lrrw[j] = 0
                        device_sum_kick_x[j] = 0
                        device_sum_kick_y[j] = 0
                        device_sum_kick_tau[j] = 0
                        if i < num_bin:
                            device_profile[i, j] = 0
                            device_density_profile[i, j] = 0
                            device_sum_bin_x[i, j] = 0
                            device_sum_bin_y[i, j] = 0
                        if i < num_bin_interp:
                            device_wp_x[i, j] = 0
                            device_wp_y[i, j] = 0
                            device_wp_tau[i, j] = 0
                        if (i < 2*num_bin_interp-1):
                            device_wt_avg[i, j] = 0
                            device_wl_avg[i, j] = 0

                else:
                    if j < num_bunch:
                        device_axis_min_tau[j] = inf
                        device_axis_max_tau[j] = -inf
                        device_num_particle_alive[j] = 0
                        device_axis_sum_x[j] = 0
                        device_axis_sum_x_squared[j] = 0
                        device_axis_sum_xp_squared[j] = 0
                        device_axis_sum_x_xp[j] = 0
                        device_axis_sum_y[j] = 0
                        device_axis_sum_y_squared[j] = 0
                        device_axis_sum_yp_squared[j] = 0
                        device_axis_sum_y_yp[j] = 0
                        device_axis_sum_tau_squared[j] = 0
                        device_axis_sum_delta_squared[j] = 0
                        device_axis_sum_tau[j] = 0
                        device_axis_sum_delta[j] = 0
                        device_axis_sum_x_lrrw[j] = 0
                        device_axis_sum_y_lrrw[j] = 0
                        device_axis_sum_tau_lrrw[j] = 0
                        device_sum_kick_x[j] = 0
                        device_sum_kick_y[j] = 0
                        device_sum_kick_tau[j] = 0
                        if i < num_bin:
                            device_profile[i, j] = 0
                            device_density_profile[i, j] = 0
                            device_sum_bin_x[i, j] = 0
                            device_sum_bin_y[i, j] = 0
                        if i < num_bin_interp:
                            device_wp_x[i, j] = 0
                            device_wp_y[i, j] = 0
                            device_wp_tau[i, j] = 0

            else:
                if k == 0:
                    if i < num_bunch:
                        device_axis_min_tau[i] = inf
                        device_axis_max_tau[i] = -inf
                        device_num_particle_alive[i] = 0
                        device_axis_sum_x[i] = 0
                        device_axis_sum_x_squared[i] = 0
                        device_axis_sum_xp_squared[i] = 0
                        device_axis_sum_x_xp[i] = 0
                        device_axis_sum_y[i] = 0
                        device_axis_sum_y_squared[i] = 0
                        device_axis_sum_yp_squared[i] = 0
                        device_axis_sum_y_yp[i] = 0
                        device_axis_sum_tau_squared[i] = 0
                        device_axis_sum_delta_squared[i] = 0
                        device_axis_sum_tau[i] = 0
                        device_axis_sum_delta[i] = 0
                        device_axis_sum_x_lrrw[i] = 0
                        device_axis_sum_y_lrrw[i] = 0
                        device_axis_sum_tau_lrrw[i] = 0
                        device_sum_kick_x[i] = 0
                        device_sum_kick_y[i] = 0
                        device_sum_kick_tau[i] = 0
                        if j < num_bin:
                            device_profile[j, i] = 0
                            device_density_profile[j, i] = 0
                            device_sum_bin_x[j, i] = 0
                            device_sum_bin_y[j, i] = 0
                        if j < num_bin_interp:
                            device_wp_x[j, i] = 0
                            device_wp_y[j, i] = 0
                            device_wp_tau[j, i] = 0
                        if (j < 2*num_bin_interp-1):
                            device_wt_avg[j, i] = 0
                            device_wl_avg[j, i] = 0

                else:
                    if i < num_bunch:
                        device_axis_min_tau[i] = inf
                        device_axis_max_tau[i] = -inf
                        device_num_particle_alive[i] = 0
                        device_axis_sum_x[i] = 0
                        device_axis_sum_x_squared[i] = 0
                        device_axis_sum_xp_squared[i] = 0
                        device_axis_sum_x_xp[i] = 0
                        device_axis_sum_y[i] = 0
                        device_axis_sum_y_squared[i] = 0
                        device_axis_sum_yp_squared[i] = 0
                        device_axis_sum_y_yp[i] = 0
                        device_axis_sum_tau_squared[i] = 0
                        device_axis_sum_delta_squared[i] = 0
                        device_axis_sum_tau[i] = 0
                        device_axis_sum_delta[i] = 0
                        device_axis_sum_x_lrrw[i] = 0
                        device_axis_sum_y_lrrw[i] = 0
                        device_axis_sum_tau_lrrw[i] = 0
                        device_sum_kick_x[i] = 0
                        device_sum_kick_y[i] = 0
                        device_sum_kick_tau[i] = 0
                        if j < num_bin:
                            device_profile[j, i] = 0
                            device_density_profile[j, i] = 0
                            device_sum_bin_x[j, i] = 0
                            device_sum_bin_y[j, i] = 0
                        if j < num_bin_interp:
                            device_wp_x[j, i] = 0
                            device_wp_y[j, i] = 0
                            device_wp_tau[j, i] = 0

        @cuda.jit
        def general_initialize_gm_kernel(device_prefix_min_tau, device_prefix_max_tau, device_axis_min_tau, device_axis_max_tau,
                                 device_axis_sum_x, device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                 device_axis_sum_y, device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                 device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau, device_axis_sum_delta,
                                 device_density_profile, device_profile, device_sum_bin_x, device_sum_bin_y, device_wl_avg,
                                 device_wtx_avg, device_wty_avg, device_wp_x, device_wp_y, device_wp_tau, device_axis_sum_x_lrrw,
                                 device_axis_sum_y_lrrw, device_axis_sum_tau_lrrw, device_sum_kick_x, device_sum_kick_y,
                                 device_sum_kick_tau, Bij, num_bunch, num_bin, num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper,
                                 device_wty_avg_upper, k):
            """
            Initialize global memory arrays

            """
            # num_particle should be larger than 2*num_bin_interp-1.
            i, j = cuda.grid(2)

            if Bij == 0:
                if k == 0:
                    if j < num_bunch:
                        device_axis_min_tau[j] = device_prefix_min_tau[0, j]
                        device_axis_max_tau[j] = device_prefix_max_tau[0, j]
                        device_axis_sum_x[j] = 0
                        device_axis_sum_x_squared[j] = 0
                        device_axis_sum_xp_squared[j] = 0
                        device_axis_sum_x_xp[j] = 0
                        device_axis_sum_y[j] = 0
                        device_axis_sum_y_squared[j] = 0
                        device_axis_sum_yp_squared[j] = 0
                        device_axis_sum_y_yp[j] = 0
                        device_axis_sum_tau_squared[j] = 0
                        device_axis_sum_delta_squared[j] = 0
                        device_axis_sum_tau[j] = 0
                        device_axis_sum_delta[j] = 0
                        device_axis_sum_x_lrrw[j] = 0
                        device_axis_sum_y_lrrw[j] = 0
                        device_axis_sum_tau_lrrw[j] = 0
                        device_sum_kick_x[j] = 0
                        device_sum_kick_y[j] = 0
                        device_sum_kick_tau[j] = 0
                        if i < num_bin:
                            device_profile[i, j] = 0
                            device_density_profile[i, j] = 0
                            device_sum_bin_x[i, j] = 0
                            device_sum_bin_y[i, j] = 0
                        if i < num_bin_interp:
                            device_wp_x[i, j] = 0
                            device_wp_y[i, j] = 0
                            device_wp_tau[i, j] = 0
                        if (i < 2*num_bin_interp-1):
                            device_wl_avg[i, j] = 0
                            device_wtx_avg[i, j] = 0
                            device_wty_avg[i, j] = 0
                            device_wl_avg_upper[i, j] = 0
                            device_wtx_avg_upper[i, j] = 0
                            device_wty_avg_upper[i, j] = 0
                else:
                    if j < num_bunch:
                        device_axis_min_tau[j] = device_prefix_min_tau[0, j]
                        device_axis_max_tau[j] = device_prefix_max_tau[0, j]
                        device_axis_sum_x[j] = 0
                        device_axis_sum_x_squared[j] = 0
                        device_axis_sum_xp_squared[j] = 0
                        device_axis_sum_x_xp[j] = 0
                        device_axis_sum_y[j] = 0
                        device_axis_sum_y_squared[j] = 0
                        device_axis_sum_yp_squared[j] = 0
                        device_axis_sum_y_yp[j] = 0
                        device_axis_sum_tau_squared[j] = 0
                        device_axis_sum_delta_squared[j] = 0
                        device_axis_sum_tau[j] = 0
                        device_axis_sum_delta[j] = 0
                        device_axis_sum_x_lrrw[j] = 0
                        device_axis_sum_y_lrrw[j] = 0
                        device_axis_sum_tau_lrrw[j] = 0
                        device_sum_kick_x[j] = 0
                        device_sum_kick_y[j] = 0
                        device_sum_kick_tau[j] = 0
                        if i < num_bin:
                            device_profile[i, j] = 0
                            device_density_profile[i, j] = 0
                            device_sum_bin_x[i, j] = 0
                            device_sum_bin_y[i, j] = 0
                        if i < num_bin_interp:
                            device_wp_x[i, j] = 0
                            device_wp_y[i, j] = 0
                            device_wp_tau[i, j] = 0

            else:
                if k == 0:
                    if i < num_bunch:
                        device_axis_min_tau[i] = device_prefix_min_tau[0, i]
                        device_axis_max_tau[i] = device_prefix_max_tau[0, i]
                        device_axis_sum_x[i] = 0
                        device_axis_sum_x_squared[i] = 0
                        device_axis_sum_xp_squared[i] = 0
                        device_axis_sum_x_xp[i] = 0
                        device_axis_sum_y[i] = 0
                        device_axis_sum_y_squared[i] = 0
                        device_axis_sum_yp_squared[i] = 0
                        device_axis_sum_y_yp[i] = 0
                        device_axis_sum_tau_squared[i] = 0
                        device_axis_sum_delta_squared[i] = 0
                        device_axis_sum_tau[i] = 0
                        device_axis_sum_delta[i] = 0
                        device_axis_sum_x_lrrw[i] = 0
                        device_axis_sum_y_lrrw[i] = 0
                        device_axis_sum_tau_lrrw[i] = 0
                        device_sum_kick_x[i] = 0
                        device_sum_kick_y[i] = 0
                        device_sum_kick_tau[i] = 0
                        if j < num_bin:
                            device_profile[j, i] = 0
                            device_density_profile[j, i] = 0
                            device_sum_bin_x[j, i] = 0
                            device_sum_bin_y[j, i] = 0
                        if j < num_bin_interp:
                            device_wp_x[j, i] = 0
                            device_wp_y[j, i] = 0
                            device_wp_tau[j, i] = 0
                        if (j < 2*num_bin_interp-1):
                            device_wl_avg[j, i] = 0
                            device_wtx_avg[j, i] = 0
                            device_wty_avg[j, i] = 0
                            device_wl_avg_upper[j, i] = 0
                            device_wtx_avg_upper[j, i] = 0
                            device_wty_avg_upper[j, i] = 0
                else:
                    if i < num_bunch:
                        device_axis_min_tau[i] = device_prefix_min_tau[0, i]
                        device_axis_max_tau[i] = device_prefix_max_tau[0, i]
                        device_axis_sum_x[i] = 0
                        device_axis_sum_x_squared[i] = 0
                        device_axis_sum_xp_squared[i] = 0
                        device_axis_sum_x_xp[i] = 0
                        device_axis_sum_y[i] = 0
                        device_axis_sum_y_squared[i] = 0
                        device_axis_sum_yp_squared[i] = 0
                        device_axis_sum_y_yp[i] = 0
                        device_axis_sum_tau_squared[i] = 0
                        device_axis_sum_delta_squared[i] = 0
                        device_axis_sum_tau[i] = 0
                        device_axis_sum_delta[i] = 0
                        device_axis_sum_x_lrrw[i] = 0
                        device_axis_sum_y_lrrw[i] = 0
                        device_axis_sum_tau_lrrw[i] = 0
                        device_sum_kick_x[i] = 0
                        device_sum_kick_y[i] = 0
                        device_sum_kick_tau[i] = 0
                        if j < num_bin:
                            device_profile[j, i] = 0
                            device_density_profile[j, i] = 0
                            device_sum_bin_x[j, i] = 0
                            device_sum_bin_y[j, i] = 0
                        if j < num_bin_interp:
                            device_wp_x[j, i] = 0
                            device_wp_y[j, i] = 0
                            device_wp_tau[j, i] = 0

        @cuda.jit
        def aperture_general_initialize_gm_kernel(inf, device_num_particle_alive, device_axis_min_tau, device_axis_max_tau,
                                 device_axis_sum_x, device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                 device_axis_sum_y, device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                 device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                 device_axis_sum_delta, device_density_profile, device_profile, device_sum_bin_x, device_sum_bin_y,
                                 device_wl_avg, device_wtx_avg, device_wty_avg, device_wp_x, device_wp_y, device_wp_tau,
                                 device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, device_axis_sum_tau_lrrw,
                                 device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau, Bij, num_bunch, num_bin, num_bin_interp,
                                 device_wl_avg_upper, device_wtx_avg_upper, device_wty_avg_upper, k):
            """
            Initialize global memory arrays

            """
            # num_particle should be larger than 2*num_bin_interp-1.
            i, j = cuda.grid(2)

            if Bij == 0:
                if k == 0:
                    if j < num_bunch:
                        device_axis_min_tau[j] = inf
                        device_axis_max_tau[j] = -inf
                        device_num_particle_alive[j] = 0
                        device_axis_sum_x[j] = 0
                        device_axis_sum_x_squared[j] = 0
                        device_axis_sum_xp_squared[j] = 0
                        device_axis_sum_x_xp[j] = 0
                        device_axis_sum_y[j] = 0
                        device_axis_sum_y_squared[j] = 0
                        device_axis_sum_yp_squared[j] = 0
                        device_axis_sum_y_yp[j] = 0
                        device_axis_sum_tau_squared[j] = 0
                        device_axis_sum_delta_squared[j] = 0
                        device_axis_sum_tau[j] = 0
                        device_axis_sum_delta[j] = 0
                        device_axis_sum_x_lrrw[j] = 0
                        device_axis_sum_y_lrrw[j] = 0
                        device_axis_sum_tau_lrrw[j] = 0
                        device_sum_kick_x[j] = 0
                        device_sum_kick_y[j] = 0
                        device_sum_kick_tau[j] = 0
                        if i < num_bin:
                            device_profile[i, j] = 0
                            device_density_profile[i, j] = 0
                            device_sum_bin_x[i, j] = 0
                            device_sum_bin_y[i, j] = 0
                        if i < num_bin_interp:
                            device_wp_x[i, j] = 0
                            device_wp_y[i, j] = 0
                            device_wp_tau[i, j] = 0
                        if (i < 2*num_bin_interp-1):
                            device_wl_avg[i, j] = 0
                            device_wtx_avg[i, j] = 0
                            device_wty_avg[i, j] = 0
                            device_wl_avg_upper[i, j] = 0
                            device_wtx_avg_upper[i, j] = 0
                            device_wty_avg_upper[i, j] = 0
                else:
                    if j < num_bunch:
                        device_axis_min_tau[j] = inf
                        device_axis_max_tau[j] = -inf
                        device_num_particle_alive[j] = 0
                        device_axis_sum_x[j] = 0
                        device_axis_sum_x_squared[j] = 0
                        device_axis_sum_xp_squared[j] = 0
                        device_axis_sum_x_xp[j] = 0
                        device_axis_sum_y[j] = 0
                        device_axis_sum_y_squared[j] = 0
                        device_axis_sum_yp_squared[j] = 0
                        device_axis_sum_y_yp[j] = 0
                        device_axis_sum_tau_squared[j] = 0
                        device_axis_sum_delta_squared[j] = 0
                        device_axis_sum_tau[j] = 0
                        device_axis_sum_delta[j] = 0
                        device_axis_sum_x_lrrw[j] = 0
                        device_axis_sum_y_lrrw[j] = 0
                        device_axis_sum_tau_lrrw[j] = 0
                        device_sum_kick_x[j] = 0
                        device_sum_kick_y[j] = 0
                        device_sum_kick_tau[j] = 0
                        if i < num_bin:
                            device_profile[i, j] = 0
                            device_density_profile[i, j] = 0
                            device_sum_bin_x[i, j] = 0
                            device_sum_bin_y[i, j] = 0
                        if i < num_bin_interp:
                            device_wp_x[i, j] = 0
                            device_wp_y[i, j] = 0
                            device_wp_tau[i, j] = 0

            else:
                if k == 0:
                    if i < num_bunch:
                        device_axis_min_tau[i] = inf
                        device_axis_max_tau[i] = -inf
                        device_num_particle_alive[i] = 0
                        device_axis_sum_x[i] = 0
                        device_axis_sum_x_squared[i] = 0
                        device_axis_sum_xp_squared[i] = 0
                        device_axis_sum_x_xp[i] = 0
                        device_axis_sum_y[i] = 0
                        device_axis_sum_y_squared[i] = 0
                        device_axis_sum_yp_squared[i] = 0
                        device_axis_sum_y_yp[i] = 0
                        device_axis_sum_tau_squared[i] = 0
                        device_axis_sum_delta_squared[i] = 0
                        device_axis_sum_tau[i] = 0
                        device_axis_sum_delta[i] = 0
                        device_axis_sum_x_lrrw[i] = 0
                        device_axis_sum_y_lrrw[i] = 0
                        device_axis_sum_tau_lrrw[i] = 0
                        device_sum_kick_x[i] = 0
                        device_sum_kick_y[i] = 0
                        device_sum_kick_tau[i] = 0
                        if j < num_bin:
                            device_profile[j, i] = 0
                            device_density_profile[j, i] = 0
                            device_sum_bin_x[j, i] = 0
                            device_sum_bin_y[j, i] = 0
                        if j < num_bin_interp:
                            device_wp_x[j, i] = 0
                            device_wp_y[j, i] = 0
                            device_wp_tau[j, i] = 0
                        if (j < 2*num_bin_interp-1):
                            device_wl_avg[j, i] = 0
                            device_wtx_avg[j, i] = 0
                            device_wty_avg[j, i] = 0
                            device_wl_avg_upper[j, i] = 0
                            device_wtx_avg_upper[j, i] = 0
                            device_wty_avg_upper[j, i] = 0
                else:
                    if i < num_bunch:
                        device_axis_min_tau[i] = inf
                        device_axis_max_tau[i] = -inf
                        device_num_particle_alive[i] = 0
                        device_axis_sum_x[i] = 0
                        device_axis_sum_x_squared[i] = 0
                        device_axis_sum_xp_squared[i] = 0
                        device_axis_sum_x_xp[i] = 0
                        device_axis_sum_y[i] = 0
                        device_axis_sum_y_squared[i] = 0
                        device_axis_sum_yp_squared[i] = 0
                        device_axis_sum_y_yp[i] = 0
                        device_axis_sum_tau_squared[i] = 0
                        device_axis_sum_delta_squared[i] = 0
                        device_axis_sum_tau[i] = 0
                        device_axis_sum_delta[i] = 0
                        device_axis_sum_x_lrrw[i] = 0
                        device_axis_sum_y_lrrw[i] = 0
                        device_axis_sum_tau_lrrw[i] = 0
                        device_sum_kick_x[i] = 0
                        device_sum_kick_y[i] = 0
                        device_sum_kick_tau[i] = 0
                        if j < num_bin:
                            device_profile[j, i] = 0
                            device_density_profile[j, i] = 0
                            device_sum_bin_x[j, i] = 0
                            device_sum_bin_y[j, i] = 0
                        if j < num_bin_interp:
                            device_wp_x[j, i] = 0
                            device_wp_y[j, i] = 0
                            device_wp_tau[j, i] = 0

        @cuda.jit
        def mm_results_kernel(Bij, num_bunch, device_prefix_min_tau, device_prefix_max_tau, device_axis_min_tau, device_axis_max_tau, num_particle_red):
            """
            min & max values for each bunch

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if 0 < i < num_particle_red and j < num_bunch:
                    cuda.atomic.min(device_axis_min_tau, j, device_prefix_min_tau[i, j])
                    cuda.atomic.max(device_axis_max_tau, j, device_prefix_max_tau[i, j])
            
            else:
                if i < num_bunch and 0 < j < num_particle_red:
                    cuda.atomic.min(device_axis_min_tau, i, device_prefix_min_tau[j, i])
                    cuda.atomic.max(device_axis_max_tau, i, device_prefix_max_tau[j, i])

        @cuda.jit
        def aperture_mm_results_kernel(Bij, num_particle, num_bunch, device_tau, device_axis_min_tau, device_axis_max_tau, device_aperture):
            """
            min & max values for each bunch when the beam loss is considered

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    if device_aperture[i, j] == 1:
                        cuda.atomic.min(device_axis_min_tau, j, device_tau[i, j]) #put inf into device_axis_min_tau[j]
                        cuda.atomic.max(device_axis_max_tau, j, device_tau[i, j]) #put -inf into device_axis_max_tau[j]
            else:
                if i < num_bunch and j < num_particle:
                    if device_aperture[j, i] == 1:
                        cuda.atomic.min(device_axis_min_tau, i, device_tau[j, i])
                        cuda.atomic.max(device_axis_max_tau, i, device_tau[j, i])
        
        @cuda.jit
        def aperture_num_particle_alive_kernel(Bij, num_particle, num_bunch, device_num_particle_alive, device_aperture):
            """
            Calculation of alive particles when the beam loss is considered

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    if device_aperture[i, j] == 1:
                        cuda.atomic.add(device_num_particle_alive, j, 1) #put 0 into device_num_particle_alive[j]
            else:
                if i < num_bunch and j < num_particle:
                    if device_aperture[j, i] == 1:
                        cuda.atomic.add(device_num_particle_alive, i, 1)

        @cuda.jit
        def binning1_kernel(num_bunch, num_bin, num_bin_interp, device_axis_min_tau, device_axis_max_tau,
                            device_axis_min_tau_interp, device_axis_max_tau_interp, device_half_d_bin_tau,
                            device_half_d_bin_tau_interp, t0, device_norm_lim_interp):
            """
            Binning kernel for resistive wall instability
            Get half_d_bin_tau

            """
            i = cuda.grid(1)
            
            if i < num_bunch:
                device_half_d_bin_tau[i] = (device_axis_max_tau[i] - device_axis_min_tau[i]) * 0.5 / (num_bin - 1)
                device_half_d_bin_tau_interp[i] = (device_axis_max_tau[i] - device_axis_min_tau[i]) * 0.5 / (num_bin_interp - 1)
                device_norm_lim_interp[i] = ( (device_axis_max_tau[i] - device_axis_min_tau[i]) * 0.5 / (num_bin_interp - 1) ) / t0
                device_axis_min_tau_interp[i] = device_axis_min_tau[i]
                device_axis_max_tau_interp[i] = device_axis_max_tau[i]

        @cuda.jit
        def binning2_kernel(num_bunch, device_axis_min_tau, device_axis_max_tau, device_axis_min_tau_interp,
                            device_axis_max_tau_interp, device_half_d_bin_tau, device_half_d_bin_tau_interp):
            """
            Binning kernel for resistive wall instability
            Patching the margins of min & max values for each bin

            """
            i = cuda.grid(1)

            if i < num_bunch:
                cuda.atomic.sub(device_axis_min_tau, i, device_half_d_bin_tau[i])
                cuda.atomic.add(device_axis_max_tau, i, device_half_d_bin_tau[i])
                cuda.atomic.sub(device_axis_min_tau_interp, i, device_half_d_bin_tau_interp[i])
                cuda.atomic.add(device_axis_max_tau_interp, i, device_half_d_bin_tau_interp[i])
        
        @cuda.jit
        def binning3_kernel(num_bunch, num_bin, num_bin_interp, device_axis_min_tau, device_axis_min_tau_interp,
                            device_bin_tau, device_bin_tau_interp, device_half_d_bin_tau, device_half_d_bin_tau_interp):
            """
            Binning kernel for resistive wall instability
            Implement binning

            """
            i = cuda.grid(1)

            if i < num_bunch:
                for idx in range(num_bin_interp):
                    device_bin_tau_interp[idx, i] = device_axis_min_tau_interp[i] + device_half_d_bin_tau_interp[i] * (2*idx + 1)
                    if idx < num_bin:
                        device_bin_tau[idx, i] = device_axis_min_tau[i] + device_half_d_bin_tau[i] * (2*idx + 1)        

        @cuda.jit
        def sorting_kernel(Bij, num_particle, num_bunch, num_bin, device_tau, device_half_d_bin_tau, device_bin_tau,
                           device_density_profile, device_profile, device_x, device_y, device_sum_bin_x,
                           device_sum_bin_y):
            """
            Sorting kernel for each bunch & calculation of zero padded charge density profile &
            partial calculation of dipole moments

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    for idx in range(num_bin):
                        if ( (device_tau[i, j] >= device_bin_tau[idx, j] - device_half_d_bin_tau[j]) and
                           (device_tau[i, j] < device_bin_tau[idx, j] + device_half_d_bin_tau[j]) ):
                            # device_charge_per_mp[j]/(2*device_half_d_bin_tau[j]*device_charge_per_bunch[j]) =
                            # 1/(2*device_half_d_bin_tau[j]*num_particle)
                            cuda.atomic.add(device_density_profile, (idx, j), 1/(2*device_half_d_bin_tau[j]*num_particle))
                            cuda.atomic.add(device_profile, (idx, j), 1)
                            cuda.atomic.add(device_sum_bin_x, (idx, j), device_x[i, j])
                            cuda.atomic.add(device_sum_bin_y, (idx, j), device_y[i, j])
                
            else:
                if i < num_bunch and j < num_particle:
                    for idx in range(num_bin):
                        if ( (device_tau[j, i] >= device_bin_tau[idx, i] - device_half_d_bin_tau[i]) and
                           (device_tau[j, i] < device_bin_tau[idx, i] + device_half_d_bin_tau[i]) ):
                            cuda.atomic.add(device_density_profile, (idx, i), 1/(2*device_half_d_bin_tau[i]*num_particle))
                            cuda.atomic.add(device_profile, (idx, i), 1)
                            cuda.atomic.add(device_sum_bin_x, (idx, i), device_x[j, i])
                            cuda.atomic.add(device_sum_bin_y, (idx, i), device_y[j, i])
        
        @cuda.jit
        def aperture_sorting_kernel(Bij, num_particle, num_bunch, num_bin, device_tau, device_half_d_bin_tau, device_bin_tau,
                           device_density_profile, device_profile, device_x, device_y, device_sum_bin_x,
                           device_sum_bin_y, device_num_particle_alive, device_aperture):
            """
            Sorting kernel for each bunch & calculation of zero padded charge density profile &
            partial calculation of dipole moments

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    for idx in range(num_bin):
                        if ( (device_tau[i, j] >= device_bin_tau[idx, j] - device_half_d_bin_tau[j]) and
                           (device_tau[i, j] < device_bin_tau[idx, j] + device_half_d_bin_tau[j]) and
                           (device_aperture[i, j] == 1) ):
                            cuda.atomic.add(device_density_profile, (idx, j), 1/(2*device_half_d_bin_tau[j]*device_num_particle_alive[j]))
                            cuda.atomic.add(device_profile, (idx, j), 1)
                            cuda.atomic.add(device_sum_bin_x, (idx, j), device_x[i, j])
                            cuda.atomic.add(device_sum_bin_y, (idx, j), device_y[i, j])
                
            else:
                if i < num_bunch and j < num_particle:
                    for idx in range(num_bin):
                        if ( (device_tau[j, i] >= device_bin_tau[idx, i] - device_half_d_bin_tau[i]) and
                           (device_tau[j, i] < device_bin_tau[idx, i] + device_half_d_bin_tau[i]) and
                           (device_aperture[j, i] == 1) ):
                            cuda.atomic.add(device_density_profile, (idx, i), 1/(2*device_half_d_bin_tau[i]*device_num_particle_alive[i]))
                            cuda.atomic.add(device_profile, (idx, i), 1)
                            cuda.atomic.add(device_sum_bin_x, (idx, i), device_x[j, i])
                            cuda.atomic.add(device_sum_bin_y, (idx, i), device_y[j, i])
                        
        @cuda.jit
        def dipole_moment_kernel(Bij, num_bunch, num_bin, device_profile, device_sum_bin_x, device_sum_bin_y,
                                 device_dip_x, device_dip_y):
            """
            Calculation of dipole moments

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_bin and j < num_bunch:
                    device_dip_x[i, j] = device_sum_bin_x[i, j] / device_profile[i, j]
                    device_dip_y[i, j] = device_sum_bin_y[i, j] / device_profile[i, j]

            else:
                if i < num_bunch and j < num_bin:
                    device_dip_x[j, i] = device_sum_bin_x[j, i] / device_profile[j, i]
                    device_dip_y[j, i] = device_sum_bin_y[j, i] / device_profile[j, i]
        
        @cuda.jit
        def nan_to_zero_kernel(Bij, num_bunch, num_bin, device_profile, device_dip_x, device_dip_y):
            """
            Convert NaN values into zeros

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_bin and j < num_bunch:
                    if device_profile[i, j] == 0:
                        device_dip_x[i, j] = 0
                        device_dip_y[i, j] = 0

            else:
                if i < num_bunch and j < num_bin:
                    if device_profile[j, i] == 0:
                        device_dip_x[j, i] = 0
                        device_dip_y[j, i] = 0

        @cuda.jit
        def density_profile_interp_kernel(Bij, num_bunch, num_bin, num_bin_interp, device_bin_tau, device_bin_tau_interp, device_density_profile,
                                          device_density_profile_interp, device_dip_x, device_dip_y, device_dip_x_interp, device_dip_y_interp):
            """
            For the wake calculations, we need to interpolate the density profile & dipole moments.

            """
            i, j = cuda.grid(2)
            
            if Bij == 0:
                if i < num_bin_interp and j < num_bunch:
                    for idx in range(num_bin-1):
                        if device_bin_tau[idx, j] <= device_bin_tau_interp[i, j] < device_bin_tau[idx+1, j]:
                            device_dip_x_interp[i, j] = ( (device_dip_x[idx+1, j] - device_dip_x[idx, j])
                                                        / (device_bin_tau[idx+1, j] - device_bin_tau[idx, j])
                                                        * (device_bin_tau_interp[i, j] - device_bin_tau[idx, j])
                                                        + device_dip_x[idx, j] )
                            device_dip_y_interp[i, j] = ( (device_dip_y[idx+1, j] - device_dip_y[idx, j])
                                                        / (device_bin_tau[idx+1, j] - device_bin_tau[idx, j])
                                                       * (device_bin_tau_interp[i, j] - device_bin_tau[idx, j])
                                                        + device_dip_y[idx, j] )
                            if idx == 0:
                                device_density_profile_interp[i, j] = ( (device_density_profile[idx+1, j] - 2*device_density_profile[idx, j])
                                                        / (device_bin_tau[idx+1, j] - device_bin_tau[idx, j])
                                                        * (device_bin_tau_interp[i, j] - device_bin_tau[idx, j])
                                                        + 2*device_density_profile[idx, j] )
                            elif idx == num_bin-2:
                                device_density_profile_interp[i, j] = ( (2*device_density_profile[idx+1, j] - device_density_profile[idx, j])
                                                        / (device_bin_tau[idx+1, j] - device_bin_tau[idx, j])
                                                        * (device_bin_tau_interp[i, j] - device_bin_tau[idx, j])
                                                        + device_density_profile[idx, j] )
                            else:
                                device_density_profile_interp[i, j] = ( (device_density_profile[idx+1, j] - device_density_profile[idx, j])
                                                        / (device_bin_tau[idx+1, j] - device_bin_tau[idx, j])
                                                        * (device_bin_tau_interp[i, j] - device_bin_tau[idx, j])
                                                        + device_density_profile[idx, j] )

                    if device_bin_tau_interp[i, j] <= device_bin_tau[0, j]:
                        device_density_profile_interp[i, j] = device_density_profile[0, j]
                        device_dip_x_interp[i, j] = device_dip_x[0, j]
                        device_dip_y_interp[i, j] = device_dip_y[0, j]
                    if device_bin_tau_interp[i, j] >= device_bin_tau[num_bin-1, j]:
                        device_density_profile_interp[i, j] = device_density_profile[num_bin-1, j]
                        device_dip_x_interp[i, j] = device_dip_x[num_bin-1, j]
                        device_dip_y_interp[i, j] = device_dip_y[num_bin-1, j]
            
            else:
                if i < num_bunch and j < num_bin_interp:
                    for idx in range(num_bin-1):
                        if device_bin_tau[idx, i] <= device_bin_tau_interp[j, i] < device_bin_tau[idx+1, i]:
                            device_dip_x_interp[j, i] = ( (device_dip_x[idx+1, i] - device_dip_x[idx, i])
                                                        / (device_bin_tau[idx+1, i] - device_bin_tau[idx, i])
                                                        * (device_bin_tau_interp[j, i] - device_bin_tau[idx, i])
                                                        + device_dip_x[idx, i] )
                            device_dip_y_interp[j, i] = ( (device_dip_y[idx+1, i] - device_dip_y[idx, i])
                                                        / (device_bin_tau[idx+1, i] - device_bin_tau[idx, i])
                                                        * (device_bin_tau_interp[j, i] - device_bin_tau[idx, i])
                                                        + device_dip_y[idx, i] )
                            if idx == 0:
                                device_density_profile_interp[j, i] = ( (device_density_profile[idx+1, i] - 2*device_density_profile[idx, i])
                                                        / (device_bin_tau[idx+1, i] - device_bin_tau[idx, i])
                                                        * (device_bin_tau_interp[j, i] - device_bin_tau[idx, i])
                                                        + 2*device_density_profile[idx, i] )
                            elif idx == num_bin-2:
                                device_density_profile_interp[j, i] = ( (2*device_density_profile[idx+1, i] - device_density_profile[idx, i])
                                                        / (device_bin_tau[idx+1, i] - device_bin_tau[idx, i])
                                                        * (device_bin_tau_interp[j, i] - device_bin_tau[idx, i])
                                                        + device_density_profile[idx, i] )
                            else:
                                device_density_profile_interp[j, i] = ( (device_density_profile[idx+1, i] - device_density_profile[idx, i])
                                                        / (device_bin_tau[idx+1, i] - device_bin_tau[idx, i])
                                                        * (device_bin_tau_interp[j, i] - device_bin_tau[idx, i])
                                                        + device_density_profile[idx, i] )

                    if device_bin_tau_interp[j, i] <= device_bin_tau[0, i]:
                        device_density_profile_interp[j, i] = device_density_profile[0, i]
                        device_dip_x_interp[j, i] = device_dip_x[0, i]
                        device_dip_y_interp[j, i] = device_dip_y[0, i]
                    if device_bin_tau_interp[j, i] >= device_bin_tau[num_bin-1, i]:
                        device_density_profile_interp[j, i] = device_density_profile[num_bin-1, i]
                        device_dip_x_interp[j, i] = device_dip_x[num_bin-1, i]
                        device_dip_y_interp[j, i] = device_dip_y[num_bin-1, i]

        @cuda.jit
        def general_wake1_kernel(Bij, num_bunch, num_wake_function, num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper, device_wty_avg_upper,
                                 device_bin_tau_interp, device_axis_min_tau_interp, device_wake_function_time, wake_function_time_interval,
                                 device_wake_function_integ_wl, device_wake_function_integ_wtx, device_wake_function_integ_wty):
            """
            1st step for Calculation of arbitrary wake functions

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_wake_function and j < num_bunch:
                    for idx in range(num_bin_interp):
                        if ( ( (device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j] >= device_wake_function_time[i, 0]) )
                            and ( (device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j] < device_wake_function_time[i, 0] 
                             + wake_function_time_interval) ) ):
                            device_wl_avg_upper[num_bin_interp-1+idx, j] = ( ((device_wake_function_integ_wl[i+1, 0] - device_wake_function_integ_wl[i, 0])
                                                                            / wake_function_time_interval
                                                                            * (device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j] - device_wake_function_time[i, 0])
                                                                            + device_wake_function_integ_wl[i, 0]) )
                            device_wtx_avg_upper[num_bin_interp-1+idx, j] = ( ((device_wake_function_integ_wtx[i+1, 0] - device_wake_function_integ_wtx[i, 0])
                                                                            / wake_function_time_interval
                                                                            * (device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j] - device_wake_function_time[i, 0])
                                                                            + device_wake_function_integ_wtx[i, 0]) )
                            device_wty_avg_upper[num_bin_interp-1+idx, j] = ( ((device_wake_function_integ_wty[i+1, 0] - device_wake_function_integ_wty[i, 0])
                                                                            / wake_function_time_interval
                                                                            * (device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j] - device_wake_function_time[i, 0])
                                                                            + device_wake_function_integ_wty[i, 0]) )
                            
            else:
                if i < num_bunch and j < num_wake_function:
                    for idx in range(num_bin_interp):
                        if ( ( (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] >= device_wake_function_time[j, 0]) )
                            and ( (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] < device_wake_function_time[j, 0] 
                             + wake_function_time_interval) ) ):
                            device_wl_avg_upper[num_bin_interp-1+idx, i] = ( ((device_wake_function_integ_wl[j+1, 0] - device_wake_function_integ_wl[j, 0])
                                                                            / wake_function_time_interval
                                                                            * (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] - device_wake_function_time[j, 0])
                                                                            + device_wake_function_integ_wl[j, 0]) )
                            device_wtx_avg_upper[num_bin_interp-1+idx, i] = ( ((device_wake_function_integ_wtx[j+1, 0] - device_wake_function_integ_wtx[j, 0])
                                                                            / wake_function_time_interval
                                                                            * (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] - device_wake_function_time[j, 0])
                                                                            + device_wake_function_integ_wtx[j, 0]) )
                            device_wty_avg_upper[num_bin_interp-1+idx, i] = ( ((device_wake_function_integ_wty[j+1, 0] - device_wake_function_integ_wty[j, 0])
                                                                            / wake_function_time_interval
                                                                            * (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] - device_wake_function_time[j, 0])
                                                                            + device_wake_function_integ_wty[j, 0]) )

        @cuda.jit
        def general_wake2_kernel(Bij, num_bunch, num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper, device_wty_avg_upper,
                                 device_wl_avg, device_wtx_avg, device_wty_avg, device_half_d_bin_tau_interp):
            """
            2nd step for Calculation of arbitrary wake functions

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if (num_bin_interp-2) < i < (2*num_bin_interp-1) and j < num_bunch:
                    device_wl_avg[i, j] = (device_wl_avg_upper[i, j] - device_wl_avg_upper[i-1, j]) / (2*device_half_d_bin_tau_interp[j])
                    device_wtx_avg[i, j] = (device_wtx_avg_upper[i, j] - device_wtx_avg_upper[i-1, j]) / (2*device_half_d_bin_tau_interp[j])
                    device_wty_avg[i, j] = (device_wty_avg_upper[i, j] - device_wty_avg_upper[i-1, j]) / (2*device_half_d_bin_tau_interp[j])
            
            else:
                if i < num_bunch and j < 2*num_bin_interp-1 and j > num_bin_interp-2:
                    device_wl_avg[j, i] = (device_wl_avg_upper[j, i] - device_wl_avg_upper[j-1, i]) / (2*device_half_d_bin_tau_interp[i])
                    device_wtx_avg[j, i] = (device_wtx_avg_upper[j, i] - device_wtx_avg_upper[j-1, i]) / (2*device_half_d_bin_tau_interp[i])
                    device_wty_avg[j, i] = (device_wty_avg_upper[j, i] - device_wty_avg_upper[j-1, i]) / (2*device_half_d_bin_tau_interp[i])
        
        # @cuda.jit
        # def general_wake1_short_long_kernel(Bij, num_bunch, num_wake_function, num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper, device_wty_avg_upper,
        #                          device_bin_tau_interp, device_axis_min_tau_interp, device_wake_function_time, wake_function_time_interval,
        #                          device_wake_function_integ_wl, device_wake_function_integ_wtx, device_wake_function_integ_wty, t_crit, amp_wl_long_integ,
        #                          amp_wt_long_integ, device_half_d_bin_tau_interp, yoklong, yokxdip, yokydip):
        #     """
        #     1st step for Calculation of arbitrary wake functions
        #     We use input wake functions for short-range calculations and built-in wake functions for long-range calculations.

        #     """
        #     i, j = cuda.grid(2)

        #     if i < num_bunch and j < num_wake_function:
        #         for idx in range(num_bin_interp):
        #             if device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] < t_crit:
        #                 if ( ( (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] >= device_wake_function_time[j, 0]) )
        #                     and ( (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] < device_wake_function_time[j, 0] 
        #                      + wake_function_time_interval) ) ):
        #                     device_wl_avg_upper[num_bin_interp-1+idx, i] = ( ((device_wake_function_integ_wl[j+1, 0] - device_wake_function_integ_wl[j, 0])
        #                                                                     / wake_function_time_interval
        #                                                                     * (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] - device_wake_function_time[j, 0])
        #                                                                     + device_wake_function_integ_wl[j, 0]) ) / (2*device_half_d_bin_tau_interp[i])
        #                     device_wtx_avg_upper[num_bin_interp-1+idx, i] = ( ((device_wake_function_integ_wtx[j+1, 0] - device_wake_function_integ_wtx[j, 0])
        #                                                                     / wake_function_time_interval
        #                                                                     * (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] - device_wake_function_time[j, 0])
        #                                                                     + device_wake_function_integ_wtx[j, 0]) ) / (2*device_half_d_bin_tau_interp[i])
        #                     device_wty_avg_upper[num_bin_interp-1+idx, i] = ( ((device_wake_function_integ_wty[j+1, 0] - device_wake_function_integ_wty[j, 0])
        #                                                                     / wake_function_time_interval
        #                                                                     * (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] - device_wake_function_time[j, 0])
        #                                                                     + device_wake_function_integ_wty[j, 0]) ) / (2*device_half_d_bin_tau_interp[i])
                    
        #             else:
        #                 # Actually, these are not upper values. They are avg values of wake functions
        #                 device_wl_avg_upper[num_bin_interp-1+idx, i] = yoklong*( ( wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
        #                                     - wl_long_integ(amp_wl_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
        #                                     / (2*device_half_d_bin_tau_interp[i]) )
        #                 device_wtx_avg_upper[num_bin_interp-1+idx, i] = yokxdip*( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
        #                                     - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
        #                                     / (2*device_half_d_bin_tau_interp[i]) )
        #                 device_wty_avg_upper[num_bin_interp-1+idx, i] = yokydip*( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
        #                                     - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
        #                                     / (2*device_half_d_bin_tau_interp[i]) )
        # @cuda.jit
        # def general_wake2_short_long_kernel(Bij, num_bunch, num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper, device_wty_avg_upper,
        #                          device_wl_avg, device_wtx_avg, device_wty_avg, t_crit, device_bin_tau_interp, device_axis_min_tau_interp):
        #     """
        #     2nd step for Calculation of arbitrary wake functions
        #     We use input wake functions for short-range calculations and built-in wake functions for long-range calculations.

        #     """
        #     i, j = cuda.grid(2)

        #     if i < num_bunch and j < 2*num_bin_interp-1 and j > num_bin_interp-2:
        #         if device_bin_tau_interp[j-num_bin_interp+1, i] - device_axis_min_tau_interp[i] < t_crit:
        #             device_wl_avg[j, i] = (device_wl_avg_upper[j, i] - device_wl_avg_upper[j-1, i])
        #             device_wtx_avg[j, i] = (device_wtx_avg_upper[j, i] - device_wtx_avg_upper[j-1, i])
        #             device_wty_avg[j, i] = (device_wty_avg_upper[j, i] - device_wty_avg_upper[j-1, i])
        #         else:
        #             device_wl_avg[j, i] = device_wl_avg_upper[j, i]
        #             device_wtx_avg[j, i] = device_wtx_avg_upper[j, i]
        #             device_wty_avg[j, i] = device_wty_avg_upper[j, i]

        @cuda.jit
        def general_wake1_short_long_kernel(Bij, num_bunch, num_wake_function, num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper, device_wty_avg_upper,
                                 device_bin_tau_interp, device_axis_min_tau_interp, device_wake_function_time, wake_function_time_interval,
                                 device_wake_function_integ_wl, device_wake_function_integ_wtx, device_wake_function_integ_wty, t_crit, amp_wl_long_integ,
                                 amp_wt_long_integ, device_half_d_bin_tau_interp, yoklong, yokxdip, yokydip):
            """
            1st step for Calculation of arbitrary wake functions
            We use input wake functions for short-range calculations and built-in wake functions for long-range calculations.

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_wake_function and j < num_bunch:
                    for idx in range(num_bin_interp):
                        if device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j] < t_crit:
                            if ( device_wake_function_time[i, 0] <= (device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j])
                             < (device_wake_function_time[i, 0] + wake_function_time_interval) ):
                                device_wl_avg_upper[num_bin_interp-1+idx, j] = ( ((device_wake_function_integ_wl[i+1, 0] - device_wake_function_integ_wl[i, 0])
                                                                        / wake_function_time_interval
                                                                        * (device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j] - device_wake_function_time[i, 0])
                                                                        + device_wake_function_integ_wl[i, 0]) ) / (2*device_half_d_bin_tau_interp[j])
                                device_wtx_avg_upper[num_bin_interp-1+idx, j] = ( ((device_wake_function_integ_wtx[i+1, 0] - device_wake_function_integ_wtx[i, 0])
                                                                        / wake_function_time_interval
                                                                        * (device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j] - device_wake_function_time[i, 0])
                                                                        + device_wake_function_integ_wtx[i, 0]) ) / (2*device_half_d_bin_tau_interp[j])
                                device_wty_avg_upper[num_bin_interp-1+idx, j] = ( ((device_wake_function_integ_wty[i+1, 0] - device_wake_function_integ_wty[i, 0])
                                                                        / wake_function_time_interval
                                                                        * (device_bin_tau_interp[idx, j] - device_axis_min_tau_interp[j] - device_wake_function_time[i, 0])
                                                                        + device_wake_function_integ_wty[i, 0]) ) / (2*device_half_d_bin_tau_interp[j])
                    
                        else:
                            # Actually, these are not upper values. They are avg values of wake functions
                            device_wl_avg_upper[num_bin_interp-1+idx, j] = yoklong*( ( wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[j]))
                                                - wl_long_integ(amp_wl_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[j])) )
                                                / (2*device_half_d_bin_tau_interp[j]) )
                            device_wtx_avg_upper[num_bin_interp-1+idx, j] = yokxdip*( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[j]))
                                                - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[j])) )
                                                / (2*device_half_d_bin_tau_interp[j]) )
                            device_wty_avg_upper[num_bin_interp-1+idx, j] = yokydip*( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[j]))
                                                - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[j])) )
                                                / (2*device_half_d_bin_tau_interp[j]) )
            
            else:
                if i < num_bunch and j < num_wake_function:
                    for idx in range(num_bin_interp):
                        if device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] < t_crit:
                            if ( device_wake_function_time[j, 0] <= (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i])
                             < (device_wake_function_time[j, 0] + wake_function_time_interval) ):
                                device_wl_avg_upper[num_bin_interp-1+idx, i] = ( ((device_wake_function_integ_wl[j+1, 0] - device_wake_function_integ_wl[j, 0])
                                                                        / wake_function_time_interval
                                                                        * (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] - device_wake_function_time[j, 0])
                                                                        + device_wake_function_integ_wl[j, 0]) ) / (2*device_half_d_bin_tau_interp[i])
                                device_wtx_avg_upper[num_bin_interp-1+idx, i] = ( ((device_wake_function_integ_wtx[j+1, 0] - device_wake_function_integ_wtx[j, 0])
                                                                        / wake_function_time_interval
                                                                        * (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] - device_wake_function_time[j, 0])
                                                                        + device_wake_function_integ_wtx[j, 0]) ) / (2*device_half_d_bin_tau_interp[i])
                                device_wty_avg_upper[num_bin_interp-1+idx, i] = ( ((device_wake_function_integ_wty[j+1, 0] - device_wake_function_integ_wty[j, 0])
                                                                        / wake_function_time_interval
                                                                        * (device_bin_tau_interp[idx, i] - device_axis_min_tau_interp[i] - device_wake_function_time[j, 0])
                                                                        + device_wake_function_integ_wty[j, 0]) ) / (2*device_half_d_bin_tau_interp[i])
                    
                        else:
                            # Actually, these are not upper values. They are avg values of wake functions
                            device_wl_avg_upper[num_bin_interp-1+idx, i] = yoklong*( ( wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                                - wl_long_integ(amp_wl_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
                                                / (2*device_half_d_bin_tau_interp[i]) )
                            device_wtx_avg_upper[num_bin_interp-1+idx, i] = yokxdip*( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                                - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
                                                / (2*device_half_d_bin_tau_interp[i]) )
                            device_wty_avg_upper[num_bin_interp-1+idx, i] = yokydip*( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                                - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
                                                / (2*device_half_d_bin_tau_interp[i]) )
                        
        @cuda.jit
        def general_wake2_short_long_kernel(Bij, num_bunch, num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper, device_wty_avg_upper,
                                 device_wl_avg, device_wtx_avg, device_wty_avg, t_crit, device_bin_tau_interp, device_axis_min_tau_interp):
            """
            2nd step for Calculation of arbitrary wake functions
            We use input wake functions for short-range calculations and built-in wake functions for long-range calculations.

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if (num_bin_interp-2) < i < (2*num_bin_interp-1) and j < num_bunch:
                    if (device_bin_tau_interp[i-num_bin_interp+1, j] - device_axis_min_tau_interp[j]) < t_crit:
                        device_wl_avg[i, j] = (device_wl_avg_upper[i, j] - device_wl_avg_upper[i-1, j])
                        device_wtx_avg[i, j] = (device_wtx_avg_upper[i, j] - device_wtx_avg_upper[i-1, j])
                        device_wty_avg[i, j] = (device_wty_avg_upper[i, j] - device_wty_avg_upper[i-1, j])
                    else:
                        device_wl_avg[i, j] = device_wl_avg_upper[i, j]
                        device_wtx_avg[i, j] = device_wtx_avg_upper[i, j]
                        device_wty_avg[i, j] = device_wty_avg_upper[i, j]

            else:
                if i < num_bunch and j < 2*num_bin_interp-1 and j > num_bin_interp-2:
                    if (device_bin_tau_interp[j-num_bin_interp+1, i] - device_axis_min_tau_interp[i]) < t_crit:
                        device_wl_avg[j, i] = (device_wl_avg_upper[j, i] - device_wl_avg_upper[j-1, i])
                        device_wtx_avg[j, i] = (device_wtx_avg_upper[j, i] - device_wtx_avg_upper[j-1, i])
                        device_wty_avg[j, i] = (device_wty_avg_upper[j, i] - device_wty_avg_upper[j-1, i])
                    else:
                        device_wl_avg[j, i] = device_wl_avg_upper[j, i]
                        device_wtx_avg[j, i] = device_wtx_avg_upper[j, i]
                        device_wty_avg[j, i] = device_wty_avg_upper[j, i]

        @cuda.jit
        def circular_rw_wake_kernel(num_bunch, num_bin_interp, amp_wl_integ, amp_wt_integ, pi, device_norm_lim_interp,
                                    device_half_d_bin_tau_interp, device_wl_avg, device_wt_avg):
            """
            Calculation of resistive wake functions by using the analytical formulas from Ivanyan and Tsakanov.
            """
            i = cuda.grid(1)

            if i < num_bunch:
                for idx in range(num_bin_interp):
                    if idx == 0:
                        wl, wt = wlwt_integ(amp_wl_integ, amp_wt_integ, pi, device_norm_lim_interp[i])
                        device_wl_avg[num_bin_interp-1+idx, i] = (wl / (2*device_half_d_bin_tau_interp[i]))
                        device_wt_avg[num_bin_interp-1+idx, i] = (wt / (2*device_half_d_bin_tau_interp[i]))
                    else:
                        wl_upper, wt_upper = wlwt_integ(amp_wl_integ, amp_wt_integ, pi, (2*idx+1)*device_norm_lim_interp[i])
                        wl_lower, wt_lower = wlwt_integ(amp_wl_integ, amp_wt_integ, pi, (2*idx-1)*device_norm_lim_interp[i])
                        device_wl_avg[num_bin_interp-1+idx, i] = ((wl_upper-wl_lower) / (2*device_half_d_bin_tau_interp[i]))
                        device_wt_avg[num_bin_interp-1+idx, i] = ((wt_upper-wt_lower) / (2*device_half_d_bin_tau_interp[i]))

        @cuda.jit
        def series_circular_rw_wake_kernel(num_bunch, num_bin_interp, t0, device_half_d_bin_tau_interp, amp_common,
                                           amp_wl_25_integ, amp_wl_long_integ, amp_wt_24_integ, amp_wt_long_integ,
                                           device_norm_lim_interp, device_wl_avg, device_wt_avg):
            """
            Calculation of resistive wake functions
            For the short-range wake, we adopt the analytical series expanded equations of Ivanyan and Tsakanov.
            We use average wake functions for each bin by integrating the given wake functions.
            Reference point for determining whether to use short-range or long-range wake function is 11.7*t0.

            """
            i = cuda.grid(1)

            if i < num_bunch:
                for idx in range(num_bin_interp):          
                    if device_half_d_bin_tau_interp[i] >= 11.7*t0:
                        if device_half_d_bin_tau_interp[i] == 11.7*t0:
                            if idx == 0:
                                device_wl_avg[num_bin_interp-1+idx, i] = ( wl_25_integ(amp_common, amp_wl_25_integ, 11.7)
                                                / (2*device_half_d_bin_tau_interp[i]) )
                                device_wt_avg[num_bin_interp-1+idx, i] = ( wt_24_integ(amp_common, amp_wt_24_integ, 11.7)
                                                / (2*device_half_d_bin_tau_interp[i]) )
                            else:
                                device_wl_avg[num_bin_interp-1+idx, i] = ( ( wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                                - wl_long_integ(amp_wl_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
                                                / (2*device_half_d_bin_tau_interp[i]) )
                                device_wt_avg[num_bin_interp-1+idx, i] = ( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                                - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
                                                / (2*device_half_d_bin_tau_interp[i]) )
                        else:
                            if idx == 0:
                                device_wl_avg[num_bin_interp-1+idx, i] = ( wl_25_integ(amp_common, amp_wl_25_integ, 11.7)
                                                + wl_long_integ(amp_wl_long_integ, device_half_d_bin_tau_interp[i])
                                                - wl_long_integ(amp_wl_long_integ, 11.7*t0)
                                                / (2*device_half_d_bin_tau_interp[i]) )
                                device_wt_avg[num_bin_interp-1+idx, i] = ( wt_24_integ(amp_common, amp_wt_24_integ, 11.7)
                                                + wt_long_integ(amp_wt_long_integ, device_half_d_bin_tau_interp[i])
                                                - wt_long_integ(amp_wt_long_integ, 11.7*t0)
                                                / (2*device_half_d_bin_tau_interp[i]) )
                            else:
                                device_wl_avg[num_bin_interp-1+idx, i] = ( ( wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                                - wl_long_integ(amp_wl_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
                                                / (2*device_half_d_bin_tau_interp[i]) )
                                device_wt_avg[num_bin_interp-1+idx, i] = ( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                                - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
                                                / (2*device_half_d_bin_tau_interp[i]) )
                    else:
                        if idx == 0:
                            device_wl_avg[num_bin_interp-1+idx, i] = ( wl_25_integ(amp_common, amp_wl_25_integ, device_norm_lim_interp[i])
                                            / (2*device_half_d_bin_tau_interp[i]) )
                            device_wt_avg[num_bin_interp-1+idx, i] = ( wt_24_integ(amp_common, amp_wt_24_integ, device_norm_lim_interp[i])
                                            / (2*device_half_d_bin_tau_interp[i]) )
                        elif 0 < idx < ( (11.7*t0+device_half_d_bin_tau_interp[i]) // (2*device_half_d_bin_tau_interp[i]) ):
                            device_wl_avg[num_bin_interp-1+idx, i] = ( ( wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx+1)*device_norm_lim_interp[i]))
                                            - wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx-1)*device_norm_lim_interp[i])) )
                                            / (2*device_half_d_bin_tau_interp[i]) )
                            device_wt_avg[num_bin_interp-1+idx, i] = ( ( wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx+1)*device_norm_lim_interp[i]))
                                            - wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx-1)*device_norm_lim_interp[i])) )
                                            / (2*device_half_d_bin_tau_interp[i]) )
                        elif idx == ( (11.7*t0+device_half_d_bin_tau_interp[i]) // (2*device_half_d_bin_tau_interp[i]) ):
                            if ( (11.7*t0+device_half_d_bin_tau_interp[i]) % (2*device_half_d_bin_tau_interp[i]) ) == 0:
                                device_wl_avg[num_bin_interp-1+idx, i] = ( ( wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx+1)*device_norm_lim_interp[i]))
                                            - wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx-1)*device_norm_lim_interp[i])) )
                                            / (2*device_half_d_bin_tau_interp[i]) )
                                device_wt_avg[num_bin_interp-1+idx, i] = ( ( wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx+1)*device_norm_lim_interp[i]))
                                            - wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx-1)*device_norm_lim_interp[i])) )
                                            / (2*device_half_d_bin_tau_interp[i]) )
                            else:
                                device_wl_avg[num_bin_interp-1+idx, i] = ( ( wl_25_integ(amp_common, amp_wl_25_integ, 11.7)
                                                - wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx-1)*device_norm_lim_interp[i]))
                                                + wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                                - wl_long_integ(amp_wl_long_integ, 11.7*t0) )
                                                / (2*device_half_d_bin_tau_interp[i]) )
                                device_wt_avg[num_bin_interp-1+idx, i] = ( ( wt_24_integ(amp_common, amp_wt_24_integ, 11.7)
                                                - wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx-1)*device_norm_lim_interp[i]))
                                                + wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                                - wt_long_integ(amp_wt_long_integ, 11.7*t0) )
                                                / (2*device_half_d_bin_tau_interp[i]) )
                        else:
                            device_wl_avg[num_bin_interp-1+idx, i] = ( ( wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                            - wl_long_integ(amp_wl_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
                                            / (2*device_half_d_bin_tau_interp[i]) )
                            device_wt_avg[num_bin_interp-1+idx, i] = ( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau_interp[i]))
                                            - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau_interp[i])) )
                                            / (2*device_half_d_bin_tau_interp[i]) )

        @cuda.jit
        def wake_convolution_kernel(Bij, num_bunch, num_bin_interp, device_wl_avg, device_wt_avg, device_wp_x, device_wp_y, device_wp_tau,
                                    device_density_profile_interp, device_dip_x_interp, device_dip_y_interp, device_half_d_bin_tau_interp):
            """
            Convolution for wakes

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if j < num_bunch:
                    for idx in range(num_bin_interp):
                        if (num_bin_interp-2) < i < (2*num_bin_interp-1):
                            cuda.atomic.add(device_wp_tau, (i - num_bin_interp + 1, j),
                                            device_wl_avg[i - idx, j] * device_density_profile_interp[idx, j] * 2*device_half_d_bin_tau_interp[j])
                            cuda.atomic.add(device_wp_x, (i - num_bin_interp + 1, j),
                                            device_wt_avg[i - idx, j] * device_density_profile_interp[idx, j] * device_dip_x_interp[idx, j]
                                            * 2*device_half_d_bin_tau_interp[j])
                            cuda.atomic.add(device_wp_y, (i - num_bin_interp + 1, j),
                                            device_wt_avg[i - idx, j] * device_density_profile_interp[idx, j] * device_dip_y_interp[idx, j]
                                            * 2*device_half_d_bin_tau_interp[j])
            
            else:
                if i < num_bunch:
                    for idx in range(num_bin_interp):
                        if (j > num_bin_interp-2) and (j < 2*num_bin_interp-1):
                            cuda.atomic.add(device_wp_tau, (j - num_bin_interp + 1, i),
                                            device_wl_avg[j - idx, i] * device_density_profile_interp[idx, i] * 2*device_half_d_bin_tau_interp[i])
                            cuda.atomic.add(device_wp_x, (j - num_bin_interp + 1, i),
                                            device_wt_avg[j - idx, i] * device_density_profile_interp[idx, i] * device_dip_x_interp[idx, i]
                                            * 2*device_half_d_bin_tau_interp[i])
                            cuda.atomic.add(device_wp_y, (j - num_bin_interp + 1, i),
                                            device_wt_avg[j - idx, i] * device_density_profile_interp[idx, i] * device_dip_y_interp[idx, i]
                                            * 2*device_half_d_bin_tau_interp[i])
        
        @cuda.jit
        def general_wake_convolution_kernel(Bij, num_bunch, num_bin_interp, device_wl_avg, device_wtx_avg, device_wty_avg, device_wp_x, device_wp_y, device_wp_tau,
                                            device_density_profile_interp, device_dip_x_interp, device_dip_y_interp, device_half_d_bin_tau_interp):
            """
            Convolution for wakes

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if j < num_bunch:
                    for idx in range(num_bin_interp):
                        if (num_bin_interp-2) < i < (2*num_bin_interp-1):
                            cuda.atomic.add(device_wp_tau, (i - num_bin_interp + 1, j),
                                            device_wl_avg[i - idx, j] * device_density_profile_interp[idx, j] * 2*device_half_d_bin_tau_interp[j])
                            cuda.atomic.add(device_wp_x, (i - num_bin_interp + 1, j),
                                            device_wtx_avg[i - idx, j] * device_density_profile_interp[idx, j] * device_dip_x_interp[idx, j]
                                            * 2*device_half_d_bin_tau_interp[j])
                            cuda.atomic.add(device_wp_y, (i - num_bin_interp + 1, j),
                                            device_wty_avg[i - idx, j] * device_density_profile_interp[idx, j] * device_dip_y_interp[idx, j]
                                            * 2*device_half_d_bin_tau_interp[j])
                            
            else:
                if i < num_bunch:
                    for idx in range(num_bin_interp):
                        if (j > num_bin_interp-2) and (j < 2*num_bin_interp-1):
                            cuda.atomic.add(device_wp_tau, (j - num_bin_interp + 1, i),
                                            device_wl_avg[j - idx, i] * device_density_profile_interp[idx, i] * 2*device_half_d_bin_tau_interp[i])
                            cuda.atomic.add(device_wp_x, (j - num_bin_interp + 1, i),
                                            device_wtx_avg[j - idx, i] * device_density_profile_interp[idx, i] * device_dip_x_interp[idx, i]
                                            * 2*device_half_d_bin_tau_interp[i])
                            cuda.atomic.add(device_wp_y, (j - num_bin_interp + 1, i),
                                            device_wty_avg[j - idx, i] * device_density_profile_interp[idx, i] * device_dip_y_interp[idx, i]
                                            * 2*device_half_d_bin_tau_interp[i])

        @cuda.jit
        def wake_interp_kernel(Bij, num_particle, num_bunch, num_bin_interp, device_wp_x, device_wp_y, device_wp_tau,
                               device_bin_tau_interp, device_tau, device_wp_x_interp, device_wp_y_interp,
                               device_wp_tau_interp):
            """
            Interpolation of wake potentials

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    for idx in range(num_bin_interp-1):
                        if (device_tau[i, j] >= device_bin_tau_interp[idx, j]) and (device_tau[i, j] < device_bin_tau_interp[idx+1, j]):
                            device_wp_tau_interp[i, j] = ( (device_wp_tau[idx+1, j] - device_wp_tau[idx, j]) / (device_bin_tau_interp[idx+1, j] - device_bin_tau_interp[idx, j])
                                                        * (device_tau[i, j] - device_bin_tau_interp[idx, j]) + device_wp_tau[idx, j] )
                            device_wp_x_interp[i, j] = ( (device_wp_x[idx+1, j] - device_wp_x[idx, j]) / (device_bin_tau_interp[idx+1, j] - device_bin_tau_interp[idx, j])
                                                   * (device_tau[i, j] - device_bin_tau_interp[idx, j]) + device_wp_x[idx, j] )
                            device_wp_y_interp[i, j] = ( (device_wp_y[idx+1, j] - device_wp_y[idx, j]) / (device_bin_tau_interp[idx+1, j] - device_bin_tau_interp[idx, j])
                                                       * (device_tau[i, j] - device_bin_tau_interp[idx, j]) + device_wp_y[idx, j] )

                    if device_tau[i, j] >= device_bin_tau_interp[num_bin_interp-1, j]:
                        device_wp_tau_interp[i, j] = device_wp_tau[num_bin_interp-1, j]
                        device_wp_x_interp[i, j] = device_wp_x[num_bin_interp-1, j]
                        device_wp_y_interp[i, j] = device_wp_y[num_bin_interp-1, j]
                    if device_tau[i, j] < device_bin_tau_interp[0, j]:
                        device_wp_tau_interp[i, j] = device_wp_tau[0, j]
                        device_wp_x_interp[i, j] = device_wp_x[0, j]
                        device_wp_y_interp[i, j] = device_wp_y[0, j]

            else:
                if i < num_bunch and j < num_particle:
                    for idx in range(num_bin_interp-1):
                        if (device_tau[j, i] >= device_bin_tau_interp[idx, i]) and (device_tau[j, i] < device_bin_tau_interp[idx+1, i]):
                            device_wp_tau_interp[j, i] = ( (device_wp_tau[idx+1, i] - device_wp_tau[idx, i]) / (device_bin_tau_interp[idx+1, i] - device_bin_tau_interp[idx, i])
                                                        * (device_tau[j, i] - device_bin_tau_interp[idx, i]) + device_wp_tau[idx, i] )
                            device_wp_x_interp[j, i] = ( (device_wp_x[idx+1, i] - device_wp_x[idx, i]) / (device_bin_tau_interp[idx+1, i] - device_bin_tau_interp[idx, i])
                                                       * (device_tau[j, i] - device_bin_tau_interp[idx, i]) + device_wp_x[idx, i] )
                            device_wp_y_interp[j, i] = ( (device_wp_y[idx+1, i] - device_wp_y[idx, i]) / (device_bin_tau_interp[idx+1, i] - device_bin_tau_interp[idx, i])
                                                       * (device_tau[j, i] - device_bin_tau_interp[idx, i]) + device_wp_y[idx, i] )

                    if device_tau[j, i] >= device_bin_tau_interp[num_bin_interp-1, i]:
                        device_wp_tau_interp[j, i] = device_wp_tau[num_bin_interp-1, i]
                        device_wp_x_interp[j, i] = device_wp_x[num_bin_interp-1, i]
                        device_wp_y_interp[j, i] = device_wp_y[num_bin_interp-1, i]
                    if device_tau[j, i] < device_bin_tau_interp[0, i]:
                        device_wp_tau_interp[j, i] = device_wp_tau[0, i]
                        device_wp_x_interp[j, i] = device_wp_x[0, i]
                        device_wp_y_interp[j, i] = device_wp_y[0, i]

        @cuda.jit
        def kick_sb_kernel(E0, Bij, num_particle, num_bunch, device_charge_per_bunch, device_wp_x_interp, device_wp_y_interp, device_wp_tau_interp,
                           device_xp, device_yp, device_delta, norm_beta):
            """
            Kick due to self-bunch wakes

            """
            i, j = cuda.grid(2)
            
            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    cuda.atomic.add(device_xp, (i, j), norm_beta[0] * device_wp_x_interp[i, j] * device_charge_per_bunch[j] / E0)
                    cuda.atomic.add(device_yp, (i, j), norm_beta[1] * device_wp_y_interp[i, j] * device_charge_per_bunch[j] / E0)
                    cuda.atomic.sub(device_delta, (i, j), device_wp_tau_interp[i, j] * device_charge_per_bunch[j] / E0)

            else:
                if i < num_bunch and j < num_particle:
                    cuda.atomic.add(device_xp, (j, i), norm_beta[0] * device_wp_x_interp[j, i] * device_charge_per_bunch[i] / E0)
                    cuda.atomic.add(device_yp, (j, i), norm_beta[1] * device_wp_y_interp[j, i] * device_charge_per_bunch[i] / E0)
                    cuda.atomic.sub(device_delta, (j, i), device_wp_tau_interp[j, i] * device_charge_per_bunch[i] / E0)

        @cuda.jit
        def aperture_kick_sb_kernel(E0, Bij, num_particle, num_bunch, device_charge_per_mp, device_num_particle_alive, device_wp_x_interp, device_wp_y_interp,
                                    device_wp_tau_interp, device_xp, device_yp, device_delta, norm_beta, device_aperture):
            """
            Kick due to self-bunch wakes

            """
            i, j = cuda.grid(2)
            
            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    if device_aperture[i, j] == 1:
                        cuda.atomic.add(device_xp, (i, j), norm_beta[0] * device_wp_x_interp[i, j] * device_num_particle_alive[j] * device_charge_per_mp[j] / E0)
                        cuda.atomic.add(device_yp, (i, j), norm_beta[1] * device_wp_y_interp[i, j] * device_num_particle_alive[j] * device_charge_per_mp[j] / E0)
                        cuda.atomic.sub(device_delta, (i, j), device_wp_tau_interp[i, j] * device_num_particle_alive[j] * device_charge_per_mp[j] / E0)

            else:
                if i < num_bunch and j < num_particle:
                    if device_aperture[j, i] == 1:
                        cuda.atomic.add(device_xp, (j, i), norm_beta[0] * device_wp_x_interp[j, i] * device_num_particle_alive[i] * device_charge_per_mp[i] / E0)
                        cuda.atomic.add(device_yp, (j, i), norm_beta[1] * device_wp_y_interp[j, i] * device_num_particle_alive[i] * device_charge_per_mp[i] / E0)
                        cuda.atomic.sub(device_delta, (j, i), device_wp_tau_interp[j, i] * device_num_particle_alive[i] * device_charge_per_mp[i] / E0)

        @cuda.jit
        def shift_tables_kernel(num_bunch, turns_lrrw, T0, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                device_tau_lrrw_roll, device_x_lrrw_roll, device_y_lrrw_roll):
            """
            Shift tables to next turn
            Table tau_lrrw is defined as the time difference of the bunch j center of mass with
            respect to center of the RF bucket number 0 at turn i.
            Turn 0 corresponds to the tracked turn.
            Positive time corresponds to past events and negative time to future events.
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)

            if i < num_bunch and j < turns_lrrw:
                # This operation corresponds to numpy.roll(array, shift=1, axis=1).
                idx = (j - 1) % turns_lrrw
                device_tau_lrrw_roll[j, i] = device_tau_lrrw[idx, i] + T0
                device_x_lrrw_roll[j, i] = device_x_lrrw[idx, i]
                device_y_lrrw_roll[j, i] = device_y_lrrw[idx, i]

        @cuda.jit
        def aperture_shift_tables_kernel(num_bunch, turns_lrrw, T0, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                         device_tau_lrrw_roll, device_x_lrrw_roll, device_y_lrrw_roll,
                                         device_charge_lrrw, device_charge_lrrw_roll):
            """
            Shift tables to next turn when the beam loss is considered
            Table tau_lrrw is defined as the time difference of the bunch j center of mass with
            respect to center of the RF bucket number 0 at turn i.
            Turn 0 corresponds to the tracked turn.
            Positive time corresponds to past events and negative time to future events.
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)

            if i < num_bunch and j < turns_lrrw:
                # This operation corresponds to numpy.roll(array, shift=1, axis=1).
                idx = (j - 1) % turns_lrrw
                device_tau_lrrw_roll[j, i] = device_tau_lrrw[idx, i] + T0
                device_x_lrrw_roll[j, i] = device_x_lrrw[idx, i]
                device_y_lrrw_roll[j, i] = device_y_lrrw[idx, i]
                device_charge_lrrw_roll[j, i] = device_charge_lrrw[idx, i]

        @cuda.jit
        def update_tables_kernel(num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                device_tau_lrrw_roll, device_x_lrrw_roll, device_y_lrrw_roll):
            """
            Update tables
            """
            i, j = cuda.grid(2)
            
            if i < num_bunch and j < turns_lrrw:
                device_tau_lrrw[j, i] = device_tau_lrrw_roll[j, i]
                device_x_lrrw[j, i] = device_x_lrrw_roll[j, i]
                device_y_lrrw[j, i] = device_y_lrrw_roll[j, i]

        @cuda.jit
        def aperture_update_tables_kernel(num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                 device_tau_lrrw_roll, device_x_lrrw_roll, device_y_lrrw_roll,
                                 device_charge_lrrw, device_charge_lrrw_roll):
            """
            Update tables
            """
            i, j = cuda.grid(2)
            
            if i < num_bunch and j < turns_lrrw:
                device_tau_lrrw[j, i] = device_tau_lrrw_roll[j, i]
                device_x_lrrw[j, i] = device_x_lrrw_roll[j, i]
                device_y_lrrw[j, i] = device_y_lrrw_roll[j, i]
                device_charge_lrrw[j, i] = device_charge_lrrw_roll[j, i]

        @cuda.jit
        def mean_ps_kernel32(device_tau, device_x, device_y, device_prefix_sum_tau_lrrw, device_prefix_sum_x_lrrw, device_prefix_sum_y_lrrw,
                            num_bunch, Jij):
            """
            Prefix sum for the calculation of mean values (tau, x, y) for each bunch
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y

            shared_sum_tau = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_x = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_y = cuda.shared.array(threadperblock, numba.float32)
            
            if Jij == 1:
                shared_sum_tau[local_i, local_j] = device_tau[i, j]
                shared_sum_x[local_i, local_j] = device_x[i, j]
                shared_sum_y[local_i, local_j] = device_y[i, j]
                cuda.syncthreads()

                s = threadperblock[0]
                s >>= 1
                while s > 0:
                    if local_i < s and local_j < threadperblock[1]:
                        shared_sum_tau[local_i, local_j] += shared_sum_tau[local_i + s, local_j]
                        shared_sum_x[local_i, local_j] += shared_sum_x[local_i + s, local_j]
                        shared_sum_y[local_i, local_j] += shared_sum_y[local_i + s, local_j]
                    cuda.syncthreads()
                    s >>= 1

                if local_i == 0 and j < num_bunch:
                    device_prefix_sum_tau_lrrw[cuda.blockIdx.x, j] = shared_sum_tau[0, local_j]
                    device_prefix_sum_x_lrrw[cuda.blockIdx.x, j] = shared_sum_x[0, local_j]
                    device_prefix_sum_y_lrrw[cuda.blockIdx.x, j] = shared_sum_y[0, local_j]

            else:
                shared_sum_tau[local_j, local_i] = device_tau[j, i]
                shared_sum_x[local_j, local_i] = device_x[j, i]
                shared_sum_y[local_j, local_i] = device_y[j, i]
                cuda.syncthreads()

                s = threadperblock[1]
                s >>= 1
                while s > 0:
                    if local_j < s and local_i < threadperblock[0]:
                        shared_sum_tau[local_j, local_i] += shared_sum_tau[local_j + s, local_i]
                        shared_sum_x[local_j, local_i] += shared_sum_x[local_j + s, local_i]
                        shared_sum_y[local_j, local_i] += shared_sum_y[local_j + s, local_i]
                    cuda.syncthreads()
                    s >>= 1

                if local_j == 0 and i < num_bunch:
                    device_prefix_sum_tau_lrrw[cuda.blockIdx.y, i] = shared_sum_tau[0, local_i]
                    device_prefix_sum_x_lrrw[cuda.blockIdx.y, i] = shared_sum_x[0, local_i]
                    device_prefix_sum_y_lrrw[cuda.blockIdx.y, i] = shared_sum_y[0, local_i]
        
        @cuda.jit
        def mean_ps_kernel64(device_tau, device_x, device_y, device_prefix_sum_tau_lrrw, device_prefix_sum_x_lrrw, device_prefix_sum_y_lrrw,
                            num_bunch, Jij):
            """
            Prefix sum for the calculation of mean values (tau, x, y) for each bunch
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y
            
            shared_sum_tau = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_x = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_y = cuda.shared.array(threadperblock, numba.float64)
            
            if Jij == 1:
                shared_sum_tau[local_i, local_j] = device_tau[i, j]
                shared_sum_x[local_i, local_j] = device_x[i, j]
                shared_sum_y[local_i, local_j] = device_y[i, j]
                cuda.syncthreads()

                s = threadperblock[0]
                s >>= 1
                while s > 0:
                    if local_i < s and local_j < threadperblock[1]:
                        shared_sum_tau[local_i, local_j] += shared_sum_tau[local_i + s, local_j]
                        shared_sum_x[local_i, local_j] += shared_sum_x[local_i + s, local_j]
                        shared_sum_y[local_i, local_j] += shared_sum_y[local_i + s, local_j]
                    cuda.syncthreads()
                    s >>= 1

                if local_i == 0 and j < num_bunch:
                    device_prefix_sum_tau_lrrw[cuda.blockIdx.x, j] = shared_sum_tau[0, local_j]
                    device_prefix_sum_x_lrrw[cuda.blockIdx.x, j] = shared_sum_x[0, local_j]
                    device_prefix_sum_y_lrrw[cuda.blockIdx.x, j] = shared_sum_y[0, local_j]

            else:
                shared_sum_tau[local_j, local_i] = device_tau[j, i]
                shared_sum_x[local_j, local_i] = device_x[j, i]
                shared_sum_y[local_j, local_i] = device_y[j, i]
                cuda.syncthreads()

                s = threadperblock[1]
                s >>= 1
                while s > 0:
                    if local_j < s and local_i < threadperblock[0]:
                        shared_sum_tau[local_j, local_i] += shared_sum_tau[local_j + s, local_i]
                        shared_sum_x[local_j, local_i] += shared_sum_x[local_j + s, local_i]
                        shared_sum_y[local_j, local_i] += shared_sum_y[local_j + s, local_i]
                    cuda.syncthreads()
                    s >>= 1

                if local_j == 0 and i < num_bunch:
                    device_prefix_sum_tau_lrrw[cuda.blockIdx.y, i] = shared_sum_tau[0, local_i]
                    device_prefix_sum_x_lrrw[cuda.blockIdx.y, i] = shared_sum_x[0, local_i]
                    device_prefix_sum_y_lrrw[cuda.blockIdx.y, i] = shared_sum_y[0, local_i]

        @cuda.jit
        def mean_as_kernel(device_prefix_sum_tau_lrrw, device_prefix_sum_x_lrrw, device_prefix_sum_y_lrrw,
                           device_axis_sum_tau_lrrw, device_axis_sum_x_lrrw, device_axis_sum_y_lrrw,
                           Bij, num_bunch, num_particle_red):
            """
            Axis sum for the calculation of mean values (tau, x, y) for each bunch
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle_red and j < num_bunch:
                    cuda.atomic.add(device_axis_sum_tau_lrrw, j, device_prefix_sum_tau_lrrw[i, j])
                    cuda.atomic.add(device_axis_sum_x_lrrw, j, device_prefix_sum_x_lrrw[i, j])
                    cuda.atomic.add(device_axis_sum_y_lrrw, j, device_prefix_sum_y_lrrw[i, j])

            else:
                if i < num_bunch and j < num_particle_red:
                    cuda.atomic.add(device_axis_sum_tau_lrrw, i, device_prefix_sum_tau_lrrw[j, i])
                    cuda.atomic.add(device_axis_sum_x_lrrw, i, device_prefix_sum_x_lrrw[j, i])
                    cuda.atomic.add(device_axis_sum_y_lrrw, i, device_prefix_sum_y_lrrw[j, i])
        
        @cuda.jit
        def aperture_mean_as_kernel(device_tau, device_x, device_y, device_axis_sum_tau_lrrw,
                                    device_axis_sum_x_lrrw, device_axis_sum_y_lrrw,
                                    Bij, num_particle, num_bunch, device_aperture):
            """
            Axis sum for the calculation of mean values (tau, x, y) for each bunch when the beam loss is considered
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    if device_aperture[i, j] == 1:
                        cuda.atomic.add(device_axis_sum_tau_lrrw, j, device_tau[i, j])
                        cuda.atomic.add(device_axis_sum_x_lrrw, j, device_x[i, j])
                        cuda.atomic.add(device_axis_sum_y_lrrw, j, device_y[i, j])
                
            else:
                if i < num_bunch and j < num_particle:
                    if device_aperture[j, i] == 1:
                        cuda.atomic.add(device_axis_sum_tau_lrrw, i, device_tau[j, i])
                        cuda.atomic.add(device_axis_sum_x_lrrw, i, device_x[j, i])
                        cuda.atomic.add(device_axis_sum_y_lrrw, i, device_y[j, i])

        @cuda.jit
        def mean_tables_kernel(device_init_long_pos, num_bunch, num_particle, device_axis_sum_tau_lrrw, device_axis_sum_x_lrrw,
                               device_axis_sum_y_lrrw, device_tau_lrrw,
                               device_x_lrrw, device_y_lrrw):
            """
            Axis sum for the calculation of mean values (tau, x, y) for each bunch
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i = cuda.grid(1)
            
            if i < num_bunch:
                device_tau_lrrw[0, i] = device_axis_sum_tau_lrrw[i] / num_particle - device_init_long_pos[i]
                device_x_lrrw[0, i] = device_axis_sum_x_lrrw[i] / num_particle
                device_y_lrrw[0, i] = device_axis_sum_y_lrrw[i] / num_particle

        @cuda.jit
        def aperture_mean_tables_kernel(device_init_long_pos, device_charge_per_mp, num_bunch, device_num_particle_alive, device_axis_sum_tau_lrrw,
                                        device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, device_tau_lrrw, device_x_lrrw,
                                        device_y_lrrw, device_charge_lrrw):
            """
            Axis sum for the calculation of mean values (tau, x, y) for each bunch
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i = cuda.grid(1)
            
            if i < num_bunch:
                device_tau_lrrw[0, i] = device_axis_sum_tau_lrrw[i] / device_num_particle_alive[i] - device_init_long_pos[i]
                device_x_lrrw[0, i] = device_axis_sum_x_lrrw[i] / device_num_particle_alive[i]
                device_y_lrrw[0, i] = device_axis_sum_y_lrrw[i] / device_num_particle_alive[i]
                device_charge_lrrw[0, i] = device_num_particle_alive[i] * device_charge_per_mp[i]

        @cuda.jit
        def get_kick_btb_kernel(num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                device_sum_kick_tau, device_sum_kick_x, device_sum_kick_y,
                                device_charge_per_bunch, amp_wl_long, amp_wt_long, yoklong, yokxdip,
                                yokydip):
            """
            Preparation of bunch to bunch kick
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)
            
            if i < num_bunch and j < turns_lrrw:
                # idx is the target bunch index.
                for idx in range(num_bunch):
                    if not j == 0 or not idx <= i:
                        cuda.atomic.add(device_sum_kick_tau, idx, (yoklong * wl_long(amp_wl_long, (device_tau_lrrw[j, i] - device_tau_lrrw[0, idx]))
                                                                  *device_charge_per_bunch[i]))
                        cuda.atomic.add(device_sum_kick_x, idx, (yokxdip * wt_long(amp_wt_long, (device_tau_lrrw[j, i] - device_tau_lrrw[0, idx]))
                                                                  *device_x_lrrw[j, i]*device_charge_per_bunch[i]))
                        cuda.atomic.add(device_sum_kick_y, idx, (yokydip * wt_long(amp_wt_long, (device_tau_lrrw[j, i] - device_tau_lrrw[0, idx]))
                                                                  *device_y_lrrw[j, i]*device_charge_per_bunch[i]))
        
        @cuda.jit
        def aperture_get_kick_btb_kernel(num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                device_sum_kick_tau, device_sum_kick_x, device_sum_kick_y,
                                device_charge_lrrw, amp_wl_long, amp_wt_long, yoklong,
                                yokxdip, yokydip):
            """
            Preparation of bunch to bunch kick
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)
            
            if i < num_bunch and j < turns_lrrw:
                # idx is the target bunch index.
                for idx in range(num_bunch):
                    if not j == 0 or not idx <= i:
                        cuda.atomic.add(device_sum_kick_tau, idx, (yoklong * wl_long(amp_wl_long, (device_tau_lrrw[j, i] - device_tau_lrrw[0, idx]))
                                                                  *device_charge_lrrw[j, i]))
                        cuda.atomic.add(device_sum_kick_x, idx, (yokxdip * wt_long(amp_wt_long, (device_tau_lrrw[j, i] - device_tau_lrrw[0, idx]))
                                                                  *device_x_lrrw[j, i]*device_charge_lrrw[j, i]))
                        cuda.atomic.add(device_sum_kick_y, idx, (yokydip * wt_long(amp_wt_long, (device_tau_lrrw[j, i] - device_tau_lrrw[0, idx]))
                                                                  *device_y_lrrw[j, i]*device_charge_lrrw[j, i]))
                            
        @cuda.jit
        def kick_btb_kernel(Bij, num_particle, num_bunch, device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau,
                            device_xp, device_yp, device_delta, E0, norm_beta):
            """
            Application of bunch to bunch kick
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)
            
            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    cuda.atomic.sub(device_delta, (i, j), device_sum_kick_tau[j] / E0)
                    cuda.atomic.add(device_xp, (i, j), norm_beta[0] * device_sum_kick_x[j] / E0)
                    cuda.atomic.add(device_yp, (i, j), norm_beta[1] * device_sum_kick_y[j] / E0)

            else:
                if i < num_bunch and j < num_particle:
                    cuda.atomic.sub(device_delta, (j, i), device_sum_kick_tau[i] / E0)
                    cuda.atomic.add(device_xp, (j, i), norm_beta[0] * device_sum_kick_x[i] / E0)
                    cuda.atomic.add(device_yp, (j, i), norm_beta[1] * device_sum_kick_y[i] / E0)

        @cuda.jit
        def monitor_ps_kernel32(device_x, device_xp, device_y, device_yp, device_tau, device_delta,
                               device_prefix_sum_x_squared, device_prefix_sum_xp_squared, device_prefix_sum_x_xp,
                               device_prefix_sum_y_squared, device_prefix_sum_yp_squared, device_prefix_sum_y_yp,
                               device_prefix_sum_tau_squared, device_prefix_sum_delta_squared, device_prefix_sum_tau,
                               device_prefix_sum_delta, device_prefix_sum_x, device_prefix_sum_y, num_bunch,
                               Jij):
            """
            Prefix sum for monitor
            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y
            
            shared_sum_x_squared = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_xp_squared = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_x_xp = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_y_squared = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_yp_squared = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_y_yp = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_tau_squared = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_delta_squared = cuda.shared.array(threadperblock, numba.float32)
            # shared_sum_tau_delta = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_tau = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_delta = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_x = cuda.shared.array(threadperblock, numba.float32)
            shared_sum_y = cuda.shared.array(threadperblock, numba.float32)
            
            if Jij == 1:
                shared_sum_x_squared[local_i, local_j] = device_x[i, j]**2
                shared_sum_xp_squared[local_i, local_j] = device_xp[i, j]**2
                shared_sum_x_xp[local_i, local_j] = device_x[i, j] * device_xp[i, j] 
                shared_sum_y_squared[local_i, local_j] = device_y[i, j]**2
                shared_sum_yp_squared[local_i, local_j] = device_yp[i, j]**2
                shared_sum_y_yp[local_i, local_j] = device_y[i, j] * device_yp[i, j]
                shared_sum_tau_squared[local_i, local_j] = device_tau[i, j]**2
                shared_sum_delta_squared[local_i, local_j] = device_delta[i, j]**2
                # shared_sum_tau_delta[local_i, local_j] = device_tau[i, j] * device_delta[i, j]
                shared_sum_tau[local_i, local_j] = device_tau[i, j]
                shared_sum_delta[local_i, local_j] = device_delta[i, j]
                shared_sum_x[local_i, local_j] = device_x[i, j]
                shared_sum_y[local_i, local_j] = device_y[i, j]
                cuda.syncthreads()

                s = threadperblock[0]
                s >>= 1
                while s > 0:
                    if local_i < s and local_j < threadperblock[1]:
                        shared_sum_x_squared[local_i, local_j] += shared_sum_x_squared[local_i + s, local_j]
                        shared_sum_xp_squared[local_i, local_j] += shared_sum_xp_squared[local_i + s, local_j]
                        shared_sum_x_xp[local_i, local_j] += shared_sum_x_xp[local_i + s, local_j]
                        shared_sum_y_squared[local_i, local_j] += shared_sum_y_squared[local_i + s, local_j]
                        shared_sum_yp_squared[local_i, local_j] += shared_sum_yp_squared[local_i + s, local_j]
                        shared_sum_y_yp[local_i, local_j] += shared_sum_y_yp[local_i + s, local_j]
                        shared_sum_tau_squared[local_i, local_j] += shared_sum_tau_squared[local_i + s, local_j]
                        shared_sum_delta_squared[local_i, local_j] += shared_sum_delta_squared[local_i + s, local_j]
                        # shared_sum_tau_delta[local_i, local_j] += shared_sum_tau_delta[local_i + s, local_j]
                        shared_sum_tau[local_i, local_j] += shared_sum_tau[local_i + s, local_j]
                        shared_sum_delta[local_i, local_j] += shared_sum_delta[local_i + s, local_j]
                        shared_sum_x[local_i, local_j] += shared_sum_x[local_i + s, local_j]
                        shared_sum_y[local_i, local_j] += shared_sum_y[local_i + s, local_j]
                    cuda.syncthreads()
                    s >>= 1
                
                if local_i == 0 and j < num_bunch:
                    device_prefix_sum_x_squared[cuda.blockIdx.x, j] = shared_sum_x_squared[0, local_j]
                    device_prefix_sum_xp_squared[cuda.blockIdx.x, j] = shared_sum_xp_squared[0, local_j]
                    device_prefix_sum_x_xp[cuda.blockIdx.x, j] = shared_sum_x_xp[0, local_j]
                    device_prefix_sum_y_squared[cuda.blockIdx.x, j] = shared_sum_y_squared[0, local_j]
                    device_prefix_sum_yp_squared[cuda.blockIdx.x, j] = shared_sum_yp_squared[0, local_j]
                    device_prefix_sum_y_yp[cuda.blockIdx.x, j] = shared_sum_y_yp[0, local_j]
                    device_prefix_sum_tau_squared[cuda.blockIdx.x, j] = shared_sum_tau_squared[0, local_j]
                    device_prefix_sum_delta_squared[cuda.blockIdx.x, j] = shared_sum_delta_squared[0, local_j]
                    # device_prefix_sum_tau_delta[cuda.blockIdx.x, j] = shared_sum_tau_delta[0, local_j]
                    device_prefix_sum_tau[cuda.blockIdx.x, j] = shared_sum_tau[0, local_j]
                    device_prefix_sum_delta[cuda.blockIdx.x, j] = shared_sum_delta[0, local_j]
                    device_prefix_sum_x[cuda.blockIdx.x, j] = shared_sum_x[0, local_j]
                    device_prefix_sum_y[cuda.blockIdx.x, j] = shared_sum_y[0, local_j]

            else:
                shared_sum_x_squared[local_j, local_i] = device_x[j, i]**2
                shared_sum_xp_squared[local_j, local_i] = device_xp[j, i]**2
                shared_sum_x_xp[local_j, local_i] = device_x[j, i] * device_xp[j, i] 
                shared_sum_y_squared[local_j, local_i] = device_y[j, i]**2
                shared_sum_yp_squared[local_j, local_i] = device_yp[j, i]**2
                shared_sum_y_yp[local_j, local_i] = device_y[j, i] * device_yp[j, i]
                shared_sum_tau_squared[local_j, local_i] = device_tau[j, i]**2
                shared_sum_delta_squared[local_j, local_i] = device_delta[j, i]**2
                # shared_sum_tau_delta[local_j, local_i] = device_tau[j, i] * device_delta[j, i]
                shared_sum_tau[local_j, local_i] = device_tau[j, i]
                shared_sum_delta[local_j, local_i] = device_delta[j, i]
                shared_sum_x[local_j, local_i] = device_x[j, i]
                shared_sum_y[local_j, local_i] = device_y[j, i]
                cuda.syncthreads()

                s = threadperblock[1]
                s >>= 1
                while s > 0:
                    if local_j < s and local_i < threadperblock[0]:
                        shared_sum_x_squared[local_j, local_i] += shared_sum_x_squared[local_j + s, local_i]
                        shared_sum_xp_squared[local_j, local_i] += shared_sum_xp_squared[local_j + s, local_i]
                        shared_sum_x_xp[local_j, local_i] += shared_sum_x_xp[local_j + s, local_i]
                        shared_sum_y_squared[local_j, local_i] += shared_sum_y_squared[local_j + s, local_i]
                        shared_sum_yp_squared[local_j, local_i] += shared_sum_yp_squared[local_j + s, local_i]
                        shared_sum_y_yp[local_j, local_i] += shared_sum_y_yp[local_j + s, local_i]
                        shared_sum_tau_squared[local_j, local_i] += shared_sum_tau_squared[local_j + s, local_i]
                        shared_sum_delta_squared[local_j, local_i] += shared_sum_delta_squared[local_j + s, local_i]
                        # shared_sum_tau_delta[local_j, local_i] += shared_sum_tau_delta[local_j + s, local_i]
                        shared_sum_tau[local_j, local_i] += shared_sum_tau[local_j + s, local_i]
                        shared_sum_delta[local_j, local_i] += shared_sum_delta[local_j + s, local_i]
                        shared_sum_x[local_j, local_i] += shared_sum_x[local_j + s, local_i]
                        shared_sum_y[local_j, local_i] += shared_sum_y[local_j + s, local_i]
                    cuda.syncthreads()
                    s >>= 1
                
                if local_j == 0 and i < num_bunch:
                    device_prefix_sum_x_squared[cuda.blockIdx.y, i] = shared_sum_x_squared[0, local_i]
                    device_prefix_sum_xp_squared[cuda.blockIdx.y, i] = shared_sum_xp_squared[0, local_i]
                    device_prefix_sum_x_xp[cuda.blockIdx.y, i] = shared_sum_x_xp[0, local_i]
                    device_prefix_sum_y_squared[cuda.blockIdx.y, i] = shared_sum_y_squared[0, local_i]
                    device_prefix_sum_yp_squared[cuda.blockIdx.y, i] = shared_sum_yp_squared[0, local_i]
                    device_prefix_sum_y_yp[cuda.blockIdx.y, i] = shared_sum_y_yp[0, local_i]
                    device_prefix_sum_tau_squared[cuda.blockIdx.y, i] = shared_sum_tau_squared[0, local_i]
                    device_prefix_sum_delta_squared[cuda.blockIdx.y, i] = shared_sum_delta_squared[0, local_i]
                    # device_prefix_sum_tau_delta[cuda.blockIdx.y, i] = shared_sum_tau_delta[0, local_i]
                    device_prefix_sum_tau[cuda.blockIdx.y, i] = shared_sum_tau[0, local_i]
                    device_prefix_sum_delta[cuda.blockIdx.y, i] = shared_sum_delta[0, local_i]
                    device_prefix_sum_x[cuda.blockIdx.y, i] = shared_sum_x[0, local_i]
                    device_prefix_sum_y[cuda.blockIdx.y, i] = shared_sum_y[0, local_i]
        
        @cuda.jit
        def monitor_ps_kernel64(device_x, device_xp, device_y, device_yp, device_tau, device_delta,
                               device_prefix_sum_x_squared, device_prefix_sum_xp_squared, device_prefix_sum_x_xp,
                               device_prefix_sum_y_squared, device_prefix_sum_yp_squared, device_prefix_sum_y_yp,
                               device_prefix_sum_tau_squared, device_prefix_sum_delta_squared, device_prefix_sum_tau,
                               device_prefix_sum_delta, device_prefix_sum_x, device_prefix_sum_y, num_bunch,
                               Jij):
            """
            Prefix sum for monitor
            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y
            
            shared_sum_x_squared = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_xp_squared = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_x_xp = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_y_squared = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_yp_squared = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_y_yp = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_tau_squared = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_delta_squared = cuda.shared.array(threadperblock, numba.float64)
            # shared_sum_tau_delta = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_tau = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_delta = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_x = cuda.shared.array(threadperblock, numba.float64)
            shared_sum_y = cuda.shared.array(threadperblock, numba.float64)
            
            if Jij == 1:
                shared_sum_x_squared[local_i, local_j] = device_x[i, j]**2
                shared_sum_xp_squared[local_i, local_j] = device_xp[i, j]**2
                shared_sum_x_xp[local_i, local_j] = device_x[i, j] * device_xp[i, j] 
                shared_sum_y_squared[local_i, local_j] = device_y[i, j]**2
                shared_sum_yp_squared[local_i, local_j] = device_yp[i, j]**2
                shared_sum_y_yp[local_i, local_j] = device_y[i, j] * device_yp[i, j]
                shared_sum_tau_squared[local_i, local_j] = device_tau[i, j]**2
                shared_sum_delta_squared[local_i, local_j] = device_delta[i, j]**2
                # shared_sum_tau_delta[local_i, local_j] = device_tau[i, j] * device_delta[i, j]
                shared_sum_tau[local_i, local_j] = device_tau[i, j]
                shared_sum_delta[local_i, local_j] = device_delta[i, j]
                shared_sum_x[local_i, local_j] = device_x[i, j]
                shared_sum_y[local_i, local_j] = device_y[i, j]
                cuda.syncthreads()

                s = threadperblock[0]
                s >>= 1
                while s > 0:
                    if local_i < s and local_j < threadperblock[1]:
                        shared_sum_x_squared[local_i, local_j] += shared_sum_x_squared[local_i + s, local_j]
                        shared_sum_xp_squared[local_i, local_j] += shared_sum_xp_squared[local_i + s, local_j]
                        shared_sum_x_xp[local_i, local_j] += shared_sum_x_xp[local_i + s, local_j]
                        shared_sum_y_squared[local_i, local_j] += shared_sum_y_squared[local_i + s, local_j]
                        shared_sum_yp_squared[local_i, local_j] += shared_sum_yp_squared[local_i + s, local_j]
                        shared_sum_y_yp[local_i, local_j] += shared_sum_y_yp[local_i + s, local_j]
                        shared_sum_tau_squared[local_i, local_j] += shared_sum_tau_squared[local_i + s, local_j]
                        shared_sum_delta_squared[local_i, local_j] += shared_sum_delta_squared[local_i + s, local_j]
                        # shared_sum_tau_delta[local_i, local_j] += shared_sum_tau_delta[local_i + s, local_j]
                        shared_sum_tau[local_i, local_j] += shared_sum_tau[local_i + s, local_j]
                        shared_sum_delta[local_i, local_j] += shared_sum_delta[local_i + s, local_j]
                        shared_sum_x[local_i, local_j] += shared_sum_x[local_i + s, local_j]
                        shared_sum_y[local_i, local_j] += shared_sum_y[local_i + s, local_j]
                    cuda.syncthreads()
                    s >>= 1
                
                if local_i == 0 and j < num_bunch:
                    device_prefix_sum_x_squared[cuda.blockIdx.x, j] = shared_sum_x_squared[0, local_j]
                    device_prefix_sum_xp_squared[cuda.blockIdx.x, j] = shared_sum_xp_squared[0, local_j]
                    device_prefix_sum_x_xp[cuda.blockIdx.x, j] = shared_sum_x_xp[0, local_j]
                    device_prefix_sum_y_squared[cuda.blockIdx.x, j] = shared_sum_y_squared[0, local_j]
                    device_prefix_sum_yp_squared[cuda.blockIdx.x, j] = shared_sum_yp_squared[0, local_j]
                    device_prefix_sum_y_yp[cuda.blockIdx.x, j] = shared_sum_y_yp[0, local_j]
                    device_prefix_sum_tau_squared[cuda.blockIdx.x, j] = shared_sum_tau_squared[0, local_j]
                    device_prefix_sum_delta_squared[cuda.blockIdx.x, j] = shared_sum_delta_squared[0, local_j]
                    # device_prefix_sum_tau_delta[cuda.blockIdx.x, j] = shared_sum_tau_delta[0, local_j]
                    device_prefix_sum_tau[cuda.blockIdx.x, j] = shared_sum_tau[0, local_j]
                    device_prefix_sum_delta[cuda.blockIdx.x, j] = shared_sum_delta[0, local_j]
                    device_prefix_sum_x[cuda.blockIdx.x, j] = shared_sum_x[0, local_j]
                    device_prefix_sum_y[cuda.blockIdx.x, j] = shared_sum_y[0, local_j]

            else:
                shared_sum_x_squared[local_j, local_i] = device_x[j, i]**2
                shared_sum_xp_squared[local_j, local_i] = device_xp[j, i]**2
                shared_sum_x_xp[local_j, local_i] = device_x[j, i] * device_xp[j, i] 
                shared_sum_y_squared[local_j, local_i] = device_y[j, i]**2
                shared_sum_yp_squared[local_j, local_i] = device_yp[j, i]**2
                shared_sum_y_yp[local_j, local_i] = device_y[j, i] * device_yp[j, i]
                shared_sum_tau_squared[local_j, local_i] = device_tau[j, i]**2
                shared_sum_delta_squared[local_j, local_i] = device_delta[j, i]**2
                # shared_sum_tau_delta[local_j, local_i] = device_tau[j, i] * device_delta[j, i]
                shared_sum_tau[local_j, local_i] = device_tau[j, i]
                shared_sum_delta[local_j, local_i] = device_delta[j, i]
                shared_sum_x[local_j, local_i] = device_x[j, i]
                shared_sum_y[local_j, local_i] = device_y[j, i]
                cuda.syncthreads()

                s = threadperblock[1]
                s >>= 1
                while s > 0:
                    if local_j < s and local_i < threadperblock[0]:
                        shared_sum_x_squared[local_j, local_i] += shared_sum_x_squared[local_j + s, local_i]
                        shared_sum_xp_squared[local_j, local_i] += shared_sum_xp_squared[local_j + s, local_i]
                        shared_sum_x_xp[local_j, local_i] += shared_sum_x_xp[local_j + s, local_i]
                        shared_sum_y_squared[local_j, local_i] += shared_sum_y_squared[local_j + s, local_i]
                        shared_sum_yp_squared[local_j, local_i] += shared_sum_yp_squared[local_j + s, local_i]
                        shared_sum_y_yp[local_j, local_i] += shared_sum_y_yp[local_j + s, local_i]
                        shared_sum_tau_squared[local_j, local_i] += shared_sum_tau_squared[local_j + s, local_i]
                        shared_sum_delta_squared[local_j, local_i] += shared_sum_delta_squared[local_j + s, local_i]
                        # shared_sum_tau_delta[local_j, local_i] += shared_sum_tau_delta[local_j + s, local_i]
                        shared_sum_tau[local_j, local_i] += shared_sum_tau[local_j + s, local_i]
                        shared_sum_delta[local_j, local_i] += shared_sum_delta[local_j + s, local_i]
                        shared_sum_x[local_j, local_i] += shared_sum_x[local_j + s, local_i]
                        shared_sum_y[local_j, local_i] += shared_sum_y[local_j + s, local_i]
                    cuda.syncthreads()
                    s >>= 1
                
                if local_j == 0 and i < num_bunch:
                    device_prefix_sum_x_squared[cuda.blockIdx.y, i] = shared_sum_x_squared[0, local_i]
                    device_prefix_sum_xp_squared[cuda.blockIdx.y, i] = shared_sum_xp_squared[0, local_i]
                    device_prefix_sum_x_xp[cuda.blockIdx.y, i] = shared_sum_x_xp[0, local_i]
                    device_prefix_sum_y_squared[cuda.blockIdx.y, i] = shared_sum_y_squared[0, local_i]
                    device_prefix_sum_yp_squared[cuda.blockIdx.y, i] = shared_sum_yp_squared[0, local_i]
                    device_prefix_sum_y_yp[cuda.blockIdx.y, i] = shared_sum_y_yp[0, local_i]
                    device_prefix_sum_tau_squared[cuda.blockIdx.y, i] = shared_sum_tau_squared[0, local_i]
                    device_prefix_sum_delta_squared[cuda.blockIdx.y, i] = shared_sum_delta_squared[0, local_i]
                    # device_prefix_sum_tau_delta[cuda.blockIdx.y, i] = shared_sum_tau_delta[0, local_i]
                    device_prefix_sum_tau[cuda.blockIdx.y, i] = shared_sum_tau[0, local_i]
                    device_prefix_sum_delta[cuda.blockIdx.y, i] = shared_sum_delta[0, local_i]
                    device_prefix_sum_x[cuda.blockIdx.y, i] = shared_sum_x[0, local_i]
                    device_prefix_sum_y[cuda.blockIdx.y, i] = shared_sum_y[0, local_i]

        @cuda.jit
        def monitor_as_kernel(device_prefix_sum_x_squared, device_prefix_sum_xp_squared, device_prefix_sum_x_xp, device_prefix_sum_y_squared,
                              device_prefix_sum_yp_squared, device_prefix_sum_y_yp, device_prefix_sum_tau_squared, device_prefix_sum_delta_squared,
                              device_prefix_sum_tau, device_prefix_sum_delta, device_prefix_sum_x, device_prefix_sum_y, device_axis_sum_x_squared,
                              device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared, device_axis_sum_yp_squared,
                              device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                              device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, Bij, num_bunch, num_particle_red):
            """
            Axis sum for monitor

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle_red and j < num_bunch:
                    cuda.atomic.add(device_axis_sum_x_squared, j, device_prefix_sum_x_squared[i, j])
                    cuda.atomic.add(device_axis_sum_xp_squared, j, device_prefix_sum_xp_squared[i, j])
                    cuda.atomic.add(device_axis_sum_x_xp, j, device_prefix_sum_x_xp[i, j])
                    cuda.atomic.add(device_axis_sum_y_squared, j, device_prefix_sum_y_squared[i, j])
                    cuda.atomic.add(device_axis_sum_yp_squared, j, device_prefix_sum_yp_squared[i, j])
                    cuda.atomic.add(device_axis_sum_y_yp, j, device_prefix_sum_y_yp[i, j])
                    cuda.atomic.add(device_axis_sum_tau_squared, j, device_prefix_sum_tau_squared[i, j])
                    cuda.atomic.add(device_axis_sum_delta_squared, j, device_prefix_sum_delta_squared[i, j])
                    # cuda.atomic.add(device_axis_sum_tau_delta, j, device_prefix_sum_tau_delta[i, j])
                    cuda.atomic.add(device_axis_sum_tau, j, device_prefix_sum_tau[i, j])
                    cuda.atomic.add(device_axis_sum_delta, j, device_prefix_sum_delta[i, j])
                    cuda.atomic.add(device_axis_sum_x, j, device_prefix_sum_x[i, j])
                    cuda.atomic.add(device_axis_sum_y, j, device_prefix_sum_y[i, j])

            else:
                if i < num_bunch and j < num_particle_red:
                    cuda.atomic.add(device_axis_sum_x_squared, i, device_prefix_sum_x_squared[j, i])
                    cuda.atomic.add(device_axis_sum_xp_squared, i, device_prefix_sum_xp_squared[j, i])
                    cuda.atomic.add(device_axis_sum_x_xp, i, device_prefix_sum_x_xp[j, i])
                    cuda.atomic.add(device_axis_sum_y_squared, i, device_prefix_sum_y_squared[j, i])
                    cuda.atomic.add(device_axis_sum_yp_squared, i, device_prefix_sum_yp_squared[j, i])
                    cuda.atomic.add(device_axis_sum_y_yp, i, device_prefix_sum_y_yp[j, i])
                    cuda.atomic.add(device_axis_sum_tau_squared, i, device_prefix_sum_tau_squared[j, i])
                    cuda.atomic.add(device_axis_sum_delta_squared, i, device_prefix_sum_delta_squared[j, i])
                    # cuda.atomic.add(device_axis_sum_tau_delta, i, device_prefix_sum_tau_delta[j, i])
                    cuda.atomic.add(device_axis_sum_tau, i, device_prefix_sum_tau[j, i])
                    cuda.atomic.add(device_axis_sum_delta, i, device_prefix_sum_delta[j, i])
                    cuda.atomic.add(device_axis_sum_x, i, device_prefix_sum_x[j, i])
                    cuda.atomic.add(device_axis_sum_y, i, device_prefix_sum_y[j, i])

        @cuda.jit
        def aperture_monitor_as_kernel(device_x, device_xp, device_y, device_yp, device_tau, device_delta, device_axis_sum_x_squared,
                                       device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared, device_axis_sum_yp_squared,
                                       device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                       device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, Bij, num_particle, num_bunch, device_aperture):
            """
            Axis sum for monitor when the beam loss is considered

            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_particle and j < num_bunch:
                    if device_aperture[i, j] == 1:
                        cuda.atomic.add(device_axis_sum_x_squared, j, device_x[i, j]**2)
                        cuda.atomic.add(device_axis_sum_xp_squared, j, device_xp[i, j]**2)
                        cuda.atomic.add(device_axis_sum_x_xp, j, device_x[i, j] * device_xp[i, j])
                        cuda.atomic.add(device_axis_sum_y_squared, j, device_y[i, j]**2)
                        cuda.atomic.add(device_axis_sum_yp_squared, j, device_yp[i, j]**2)
                        cuda.atomic.add(device_axis_sum_y_yp, j, device_y[i, j] * device_yp[i, j])
                        cuda.atomic.add(device_axis_sum_tau_squared, j, device_tau[i, j]**2)
                        cuda.atomic.add(device_axis_sum_delta_squared, j, device_delta[i, j]**2)
                        # cuda.atomic.add(device_axis_sum_tau_delta, j, device_tau[i, j] * device_delta[i, j])
                        cuda.atomic.add(device_axis_sum_tau, j, device_tau[i, j])
                        cuda.atomic.add(device_axis_sum_delta, j, device_delta[i, j])
                        cuda.atomic.add(device_axis_sum_x, j, device_x[i, j])
                        cuda.atomic.add(device_axis_sum_y, j, device_y[i, j])

            else:
                if i < num_bunch and j < num_particle:
                    if device_aperture[j, i] == 1:
                        cuda.atomic.add(device_axis_sum_x_squared, i, device_x[j, i]**2)
                        cuda.atomic.add(device_axis_sum_xp_squared, i, device_xp[j, i]**2)
                        cuda.atomic.add(device_axis_sum_x_xp, i, device_x[j, i] * device_xp[j, i])
                        cuda.atomic.add(device_axis_sum_y_squared, i, device_y[j, i]**2)
                        cuda.atomic.add(device_axis_sum_yp_squared, i, device_yp[j, i]**2)
                        cuda.atomic.add(device_axis_sum_y_yp, i, device_y[j, i] * device_yp[j, i])
                        cuda.atomic.add(device_axis_sum_tau_squared, i, device_tau[j, i]**2)
                        cuda.atomic.add(device_axis_sum_delta_squared, i, device_delta[j, i]**2)
                        # cuda.atomic.add(device_axis_sum_tau_delta, i, device_tau[j, i] * device_delta[j, i])
                        cuda.atomic.add(device_axis_sum_tau, i, device_tau[j, i])
                        cuda.atomic.add(device_axis_sum_delta, i, device_delta[j, i])
                        cuda.atomic.add(device_axis_sum_x, i, device_x[j, i])
                        cuda.atomic.add(device_axis_sum_y, i, device_y[j, i])

        @cuda.jit
        def monitor_results_kernel(device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared,
                                   device_axis_sum_yp_squared, device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared,
                                   device_axis_sum_tau, device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, device_bunch_length,
                                   device_energy_spread, device_Jx, device_Jy, device_beam_sizeX, device_beam_sizeY, device_beam_comS,
                                   device_beam_comX, device_beam_comY, alpha_x, alpha_y, beta_x, beta_y, gamma_x, gamma_y,
                                   num_bunch, num_particle, k):
            """
            Final results

            """
            i = cuda.grid(1)
            
            if i < num_bunch:
                # device_beam_emitX[k+1, i] = sqrt( (device_axis_sum_x_squared[i] * device_axis_sum_xp_squared[i] - device_axis_sum_x_xp[i]**2) ) / num_particle
                # device_beam_emitY[k+1, i] = sqrt( (device_axis_sum_y_squared[i] * device_axis_sum_yp_squared[i] - device_axis_sum_y_yp[i]**2) ) / num_particle
                # device_beam_emitS[k+1, i] = sqrt( (device_axis_sum_tau_squared[i] * device_axis_sum_delta_squared[i] - device_axis_sum_tau_delta[i]**2) ) / num_particle
                device_bunch_length[k+1, i] = sqrt( (device_axis_sum_tau_squared[i]/num_particle) - (device_axis_sum_tau[i]/num_particle)**2 )
                device_energy_spread[k+1, i] = sqrt( (device_axis_sum_delta_squared[i]/num_particle) - (device_axis_sum_delta[i]/num_particle)**2 )
                device_Jx[k+1, i] = ( gamma_x*device_axis_sum_x_squared[i] + 2*alpha_x*device_axis_sum_x_xp[i] + beta_x*device_axis_sum_xp_squared[i] ) / num_particle
                device_Jy[k+1, i] = ( gamma_y*device_axis_sum_y_squared[i] + 2*alpha_y*device_axis_sum_y_yp[i] + beta_y*device_axis_sum_yp_squared[i] ) / num_particle
                device_beam_sizeX[k+1, i] = sqrt( (device_axis_sum_x_squared[i]/num_particle) - (device_axis_sum_x[i]/num_particle)**2 )
                device_beam_sizeY[k+1, i] = sqrt( (device_axis_sum_y_squared[i]/num_particle) - (device_axis_sum_y[i]/num_particle)**2 )
                device_beam_comS[k+1, i] = device_axis_sum_tau[i]/num_particle
                device_beam_comX[k+1, i] = device_axis_sum_x[i]/num_particle
                device_beam_comY[k+1, i] = device_axis_sum_y[i]/num_particle

        @cuda.jit
        def aperture_monitor_results_kernel(device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared,
                                            device_axis_sum_yp_squared, device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared,
                                            device_axis_sum_tau, device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, device_bunch_length,
                                            device_energy_spread, device_Jx, device_Jy, device_beam_sizeX, device_beam_sizeY, device_beam_comS,
                                            device_beam_comX, device_beam_comY, alpha_x, alpha_y, beta_x, beta_y, gamma_x, gamma_y,
                                            num_bunch, num_particle, device_num_particle_alive, k):
            """
            Final results when the beam loss is considered

            """
            i = cuda.grid(1)
            
            if i < num_bunch:
                # device_beam_emitX[k+1, i] = sqrt( (device_axis_sum_x_squared[i] * device_axis_sum_xp_squared[i] - device_axis_sum_x_xp[i]**2) ) / device_num_particle_alive[i]
                # device_beam_emitY[k+1, i] = sqrt( (device_axis_sum_y_squared[i] * device_axis_sum_yp_squared[i] - device_axis_sum_y_yp[i]**2) ) / device_num_particle_alive[i]
                # device_beam_emitS[k+1, i] = sqrt( (device_axis_sum_tau_squared[i] * device_axis_sum_delta_squared[i] - device_axis_sum_tau_delta[i]**2) ) / device_num_particle_alive[i]
                device_bunch_length[k+1, i] = sqrt( (device_axis_sum_tau_squared[i]/device_num_particle_alive[i]) - (device_axis_sum_tau[i]/device_num_particle_alive[i])**2 )
                device_energy_spread[k+1, i] = sqrt( (device_axis_sum_delta_squared[i]/device_num_particle_alive[i]) - (device_axis_sum_delta[i]/device_num_particle_alive[i])**2 )
                device_Jx[k+1, i] = ( gamma_x*device_axis_sum_x_squared[i] + 2*alpha_x*device_axis_sum_x_xp[i] + beta_x*device_axis_sum_xp_squared[i] ) / device_num_particle_alive[i]
                device_Jy[k+1, i] = ( gamma_y*device_axis_sum_y_squared[i] + 2*alpha_y*device_axis_sum_y_yp[i] + beta_y*device_axis_sum_yp_squared[i] ) / device_num_particle_alive[i]
                device_beam_sizeX[k+1, i] = sqrt( (device_axis_sum_x_squared[i]/device_num_particle_alive[i]) - (device_axis_sum_x[i]/device_num_particle_alive[i])**2 )
                device_beam_sizeY[k+1, i] = sqrt( (device_axis_sum_y_squared[i]/device_num_particle_alive[i]) - (device_axis_sum_y[i]/device_num_particle_alive[i])**2 )
                device_beam_comS[k+1, i] = device_axis_sum_tau[i]/num_particle
                device_beam_comX[k+1, i] = device_axis_sum_x[i]/num_particle
                device_beam_comY[k+1, i] = device_axis_sum_y[i]/num_particle
        
        @cuda.jit
        def monitor_dipole_moments_kernel(device_dip_x, device_dip_y, device_dip_x_turns, device_dip_y_turns, Bij, num_bunch, num_bin, k):
            """
            Monitors for dipole moments
            """
            i, j = cuda.grid(2)

            if Bij == 0:
                if i < num_bin and j < num_bunch:
                    device_dip_x_turns[k, i, j] = device_dip_x[i, j]
                    device_dip_y_turns[k, i, j] = device_dip_y[i, j]

            else:
                if i < num_bunch and j < num_bin:
                    device_dip_x_turns[k, j, i] = device_dip_x[j, i]
                    device_dip_y_turns[k, j, i] = device_dip_y[j, i]

        if isinstance(bunch, Beam):
            dtype = "f8" if dblPrec6D else "f4"
            dtype_gpu = np.float64 if dblPrec6D else np.float32
            beam = bunch
            num_bunch = len([fp for fp in self.filling_pattern if fp != 0])
            num_particle = beam[0].mp_number # zeroth bunch must be filled.
            charge_per_bunch = np.empty((num_bunch), dtype=dtype)
            charge_per_mp = np.empty((num_bunch), dtype=dtype)
            if culrrw:
                init_long_pos_uni = np.arange(self.ring.h) * self.ring.T1
                init_long_pos = np.empty((num_bunch), dtype=dtype)
                tau_lrrw = np.ones((turns_lrrw, num_bunch), dtype=dtype) * inf
                x_lrrw = np.zeros((turns_lrrw, num_bunch), dtype=dtype)
                y_lrrw = np.zeros((turns_lrrw, num_bunch), dtype=dtype)

                if rectangularaperture:
                    charge_lrrw = np.zeros((turns_lrrw, num_bunch), dtype=dtype)
            
            x = np.empty((num_particle, num_bunch), dtype=dtype)
            xp = np.empty((num_particle, num_bunch), dtype=dtype)
            y = np.empty((num_particle, num_bunch), dtype=dtype)
            yp = np.empty((num_particle, num_bunch), dtype=dtype)
            tau = np.empty((num_particle, num_bunch), dtype=dtype)
            delta = np.empty((num_particle, num_bunch), dtype=dtype)
            tau_save = np.empty((num_particle), dtype=dtype)
            
            if not BunchToJ:
                Jij = 0
            if BunchToJ:
                Jij = 1
            
            if not BunchToX:
                Bij = 0
            if BunchToX:
                Bij = 1
                Jij = 0

            if curm:
                # beam_emitX = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                # beam_emitY = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                # beam_emitS = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                bunch_length = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                energy_spread = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                Jx = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                Jy = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                beam_sizeX = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                beam_sizeY = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                beam_comS = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                beam_comX = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                beam_comY = np.empty((int(turns/curm_ti)+1, num_bunch), dtype=dtype)
                if monitordip:
                    dipX = np.empty((int(turns/curm_ti), self.num_bin, num_bunch), dtype=dtype)
                    dipY = np.empty((int(turns/curm_ti), self.num_bin, num_bunch), dtype=dtype)

            if not curm:
                # beam_emitX = np.empty((turns+1, num_bunch), dtype=dtype)
                # beam_emitY = np.empty((turns+1, num_bunch), dtype=dtype
                # beam_emitS = np.empty((turns+1, num_bunch), dtype=dtype)
                bunch_length = np.empty((turns+1, num_bunch), dtype=dtype)
                energy_spread = np.empty((turns+1, num_bunch), dtype=dtype)
                Jx = np.empty((turns+1, num_bunch), dtype=dtype)
                Jy = np.empty((turns+1, num_bunch), dtype=dtype)
                beam_sizeX = np.empty((turns+1, num_bunch), dtype=dtype)
                beam_sizeY = np.empty((turns+1, num_bunch), dtype=dtype)
                beam_comS = np.empty((turns+1, num_bunch), dtype=dtype)
                beam_comX = np.empty((turns+1, num_bunch), dtype=dtype)
                beam_comY = np.empty((turns+1, num_bunch), dtype=dtype)
                if monitordip:
                    dipX = np.empty((turns, self.num_bin, num_bunch), dtype=dtype)
                    dipY = np.empty((turns, self.num_bin, num_bunch), dtype=dtype)

            # sigma_x = self.ring.sigma()[0]
            sigma_xp = self.ring.sigma()[1]
            # sigma_y = self.ring.sigma()[2]
            sigma_yp = self.ring.sigma()[3]
            tau_h = self.ring.tau[0]
            tau_v = self.ring.tau[1]
            tau_l = self.ring.tau[2]
            tune_x = self.ring.tune[0]
            tune_y = self.ring.tune[1]
            chro_x = self.ring.chro[0]
            chro_y = self.ring.chro[1]
            alpha_x = self.alpha[0]
            alpha_y = self.alpha[1]
            beta_x = self.beta[0]
            beta_y = self.beta[1]
            gamma_x = self.gamma[0]
            gamma_y = self.gamma[1]
            dispersion_x = self.dispersion[0]
            dispersion_xp = self.dispersion[1]
            dispersion_y = self.dispersion[2]
            dispersion_yp = self.dispersion[3]

            Z0 = mu_0*c
            t0 = (2*self.rho*self.radius_y**2/Z0)**(1/3) / c

            if 11.7*t0 > self.ring.T1:
                raise ValueError("The approximated wake functions are not valid.")

            if cuelliptic:
                yoklong, yokxdip, yokydip, yokxquad, yokyquad = yokoya_elliptic(self.radius_x, self.radius_y)
            if not cuelliptic:
                yoklong, yokxdip, yokydip = np.ones(3)
                yokxquad, yokyquad = np.zeros(2)
            
            if cugeneralwake:
                num_wake_function = len(self.wake_function_time)
                wake_function_time_interval = self.wake_function_time[1] - self.wake_function_time[0]
                wake_function_time2 = np.empty((num_wake_function, 1), dtype=dtype)
                wake_function_time2[:, 0] = self.wake_function_time
                wake_function_integ_wl2 = np.empty((num_wake_function, 1), dtype=dtype)
                wake_function_integ_wl2[:, 0] = self.wake_function_integ_wl
                wake_function_integ_wtx2 = np.empty((num_wake_function, 1), dtype=dtype)
                wake_function_integ_wtx2[:, 0] = self.wake_function_integ_wtx
                wake_function_integ_wty2 = np.empty((num_wake_function, 1), dtype=dtype)
                wake_function_integ_wty2[:, 0] = self.wake_function_integ_wty
            
            if rectangularaperture:
                aperture = np.ones((num_particle, num_bunch), dtype=dtype)
                num_particle_alive = np.empty(num_bunch, dtype=dtype)

            amp_common = 0.5*sqrt(2/pi)
            amp_wl_25_integ = (Z0*c*t0) / (pi*self.radius_y**2) * self.length
            amp_wl_long_integ = sqrt(Z0 * self.rho / (c * pi)) / (2*pi*self.radius_y) * self.length
            amp_wt_24_integ = (Z0*c**2*t0**2) / (pi*self.radius_y**4) * self.length
            amp_wt_long_integ = 2 * sqrt(Z0*c*self.rho / pi) / (pi*self.radius_y**3) * self.length
            
            amp_wl_integ = (Z0*c*t0) / (3*pi*self.radius_y**2) * self.length
            amp_wt_integ = (Z0*c**2*t0**2) / (3*pi*self.radius_y**4) * self.length
            amp_wl_long = -1 * sqrt(Z0*self.rho / (c*pi)) / (4*pi*self.radius_y) * self.length
            amp_wt_long = sqrt(Z0*c*self.rho / pi) / (pi*self.radius_y**3) * self.length
            
            # if not cugeneralwake:
            #     amp_wl_long = -1 * sqrt(Z0*self.rho / (c*pi)) / (4*pi*self.radius_y) * self.length
            #     amp_wt_long = sqrt(Z0*c*self.rho / pi) / (pi*self.radius_y**3) * self.length
            
            # if cugeneralwake:
            #     amp_wl_long = -1 * sqrt(Z0*self.rho / (c*pi)) / (4*pi*self.r_lrrw) * self.length
            #     amp_wtx_long = sqrt(Z0*c*self.rho / pi) / (pi*self.x3_lrrw**3) * self.length
            #     amp_wty_long = sqrt(Z0*c*self.rho / pi) / (pi*self.y3_lrrw**3) * self.length

            non_zero_indices = [idx_fp for idx_fp, val_fp in enumerate(self.filling_pattern) if val_fp != 0]
            for idx_bunch in range(num_bunch):
                x[:, idx_bunch] = beam[non_zero_indices[idx_bunch]].particles["x"]
                xp[:, idx_bunch] = beam[non_zero_indices[idx_bunch]].particles["xp"]
                y[:, idx_bunch] = beam[non_zero_indices[idx_bunch]].particles["y"]
                yp[:, idx_bunch] = beam[non_zero_indices[idx_bunch]].particles["yp"]
                tau[:, idx_bunch] = beam[non_zero_indices[idx_bunch]].particles["tau"]
                delta[:, idx_bunch] = beam[non_zero_indices[idx_bunch]].particles["delta"]
                # beam_emitX[0, idx_bunch] = beam[non_zero_indices[idx_bunch]].emit[0]
                # beam_emitY[0, idx_bunch] = beam[non_zero_indices[idx_bunch]].emit[1]
                # beam_emitS[0, idx_bunch] = beam[non_zero_indices[idx_bunch]].emit[2]

                bunch_length[0, idx_bunch] = beam[non_zero_indices[idx_bunch]].std[4]
                energy_spread[0, idx_bunch] = beam[non_zero_indices[idx_bunch]].std[5]
                Jx[0, idx_bunch] = beam[non_zero_indices[idx_bunch]].cs_invariant[0]
                Jy[0, idx_bunch] = beam[non_zero_indices[idx_bunch]].cs_invariant[1]
                beam_sizeX[0, idx_bunch] = beam[non_zero_indices[idx_bunch]].std[0]
                beam_sizeY[0, idx_bunch] = beam[non_zero_indices[idx_bunch]].std[2]
                beam_comS[0, idx_bunch] = 0
                beam_comX[0, idx_bunch] = 0
                beam_comY[0, idx_bunch] = 0
                charge_per_bunch[idx_bunch] = beam[non_zero_indices[idx_bunch]].charge
                charge_per_mp[idx_bunch] = beam[non_zero_indices[idx_bunch]].charge_per_mp

                if culrrw:
                    init_long_pos[idx_bunch] = init_long_pos_uni[non_zero_indices[idx_bunch]]

            threadperblock = (self.thread_per_block, self.thread_per_block)
            if not BunchToX:
                blockpergrid = (ceil(num_particle / self.thread_per_block), ceil(num_bunch / self.thread_per_block))
                num_particle_red = blockpergrid[0]
                num_bunch_red = blockpergrid[1]
                if not BunchToJ:
                    blockpergrid_par = (num_bunch_red, num_particle_red)
                if BunchToJ:
                    blockpergrid_par = blockpergrid
                blockpergrid_red = (ceil(num_particle_red / self.thread_per_block), num_bunch_red)
                blockpergrid_pad = (ceil((2*self.num_bin_interp-1) / self.thread_per_block), num_bunch_red)
                blockpergrid_bin = (ceil(self.num_bin / self.thread_per_block), num_bunch_red)
                blockpergrid_bin_interp = (ceil(self.num_bin_interp / self.thread_per_block), num_bunch_red)
                if cugeneralwake:
                    blockpergrid_wake = (ceil(num_wake_function / self.thread_per_block), num_bunch_red)
                if culrrw:
                    blockpergrid_lrrw = (num_bunch_red, ceil(turns_lrrw / self.thread_per_block))
            if BunchToX:
                blockpergrid = (ceil(num_bunch / self.thread_per_block), ceil(num_particle / self.thread_per_block))
                num_particle_red = blockpergrid[1]
                num_bunch_red = blockpergrid[0]
                blockpergrid_par = blockpergrid
                blockpergrid_red = (num_bunch_red, ceil(num_particle_red / self.thread_per_block))
                blockpergrid_pad = (num_bunch_red, ceil((2*self.num_bin_interp-1) / self.thread_per_block))
                blockpergrid_bin = (num_bunch_red, ceil(self.num_bin / self.thread_per_block))
                blockpergrid_bin_interp = (num_bunch_red, ceil(self.num_bin_interp / self.thread_per_block))
                if cugeneralwake:
                    blockpergrid_wake = (num_bunch_red, ceil(num_wake_function / self.thread_per_block))
                if culrrw:
                    blockpergrid_lrrw = (num_bunch_red, ceil(turns_lrrw / self.thread_per_block))

            seed1 = os.getpid()

            # Calculations in GPU
                
            # Create a CUDA stream
            stream = cuda.stream()

            if cusr:
                rng_states1 = create_xoroshiro128p_states(3*(num_particle+turns-1), seed=seed1, stream=stream)
                # rng_states1 = create_xoroshiro128p_states(num_particle*turns, seed=seed1, stream=stream)
            
            # device_beam_emitX = cuda.to_device(beam_emitX, stream=stream)
            # device_beam_emitY = cuda.to_device(beam_emitY, stream=stream)
            # device_beam_emitS = cuda.to_device(beam_emitS, stream=stream)

            device_bunch_length = cuda.to_device(bunch_length, stream=stream)
            device_energy_spread = cuda.to_device(energy_spread, stream=stream)
            device_Jx = cuda.to_device(Jx, stream=stream)
            device_Jy = cuda.to_device(Jy, stream=stream)
            device_beam_sizeX = cuda.to_device(beam_sizeX, stream=stream)
            device_beam_sizeY = cuda.to_device(beam_sizeY, stream=stream)
            device_beam_comS = cuda.to_device(beam_comS, stream=stream)
            device_beam_comX = cuda.to_device(beam_comX, stream=stream)
            device_beam_comY = cuda.to_device(beam_comY, stream=stream)
            
            if monitordip:
                device_dipX = cuda.to_device(dipX, stream=stream)
                device_dipY = cuda.to_device(dipY, stream=stream)

            if cugeneralwake:
                device_wake_function_time = cuda.to_device(wake_function_time2, stream=stream)
                device_wake_function_integ_wl = cuda.to_device(wake_function_integ_wl2, stream=stream)
                device_wake_function_integ_wtx = cuda.to_device(wake_function_integ_wtx2, stream=stream)
                device_wake_function_integ_wty = cuda.to_device(wake_function_integ_wty2, stream=stream)
            
            if rectangularaperture:
                device_aperture = cuda.to_device(aperture, stream=stream)
                device_num_particle_alive = cuda.device_array_like(num_particle_alive, stream=stream)
                if culrrw:
                    device_charge_lrrw = cuda.to_device(charge_lrrw, stream=stream)
                    device_charge_lrrw_roll = cuda.device_array_like(charge_lrrw, stream=stream)

            device_x = cuda.to_device(x, stream=stream)
            device_xp = cuda.to_device(xp, stream=stream)
            device_y = cuda.to_device(y, stream=stream)
            device_yp = cuda.to_device(yp, stream=stream)
            device_tau = cuda.to_device(tau, stream=stream)
            device_delta = cuda.to_device(delta, stream=stream)

            device_charge_per_bunch = cuda.to_device(charge_per_bunch, stream=stream)

            if culrrw:
                device_charge_per_mp = cuda.to_device(charge_per_mp, stream=stream)
                device_init_long_pos = cuda.to_device(init_long_pos, stream=stream)

                device_x_lrrw = cuda.to_device(x_lrrw, stream=stream)
                device_y_lrrw = cuda.to_device(y_lrrw, stream=stream)
                device_tau_lrrw = cuda.to_device(tau_lrrw, stream=stream)

                device_x_lrrw_roll = cuda.device_array_like(x_lrrw, stream=stream)
                device_y_lrrw_roll = cuda.device_array_like(y_lrrw, stream=stream)
                device_tau_lrrw_roll = cuda.device_array_like(tau_lrrw, stream=stream)

            device_xp_sr = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)
            device_yp_sr = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)
            device_delta_sr = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)
            device_x_tm = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)
            device_xp_tm = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)
            device_y_tm = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)
            device_yp_tm = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)

            device_rand_xp = cuda.device_array((num_particle,), dtype=dtype_gpu, stream=stream)
            device_rand_yp = cuda.device_array((num_particle,), dtype=dtype_gpu, stream=stream)
            device_rand_delta = cuda.device_array((num_particle,), dtype=dtype_gpu, stream=stream)

            device_prefix_min_tau = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_max_tau = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
                
            device_axis_min_tau = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_max_tau = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_min_tau_interp = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_max_tau_interp = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)

            device_bin_tau = cuda.device_array((self.num_bin, num_bunch), dtype=dtype_gpu, stream=stream)
            device_bin_tau_interp = cuda.device_array((self.num_bin_interp, num_bunch), dtype=dtype_gpu, stream=stream)
            device_half_d_bin_tau = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_half_d_bin_tau_interp = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            # device_t = cuda.device_array((self.num_bin, num_bunch), dtype=dtype_gpu, stream=stream)
            device_norm_lim_interp = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)

            device_wl_avg = cuda.device_array((2*self.num_bin_interp-1, num_bunch), dtype=dtype_gpu, stream=stream)

            # wp_tau_interp = np.empty((num_particle, num_bunch), dtype=dtype)
            # wp_x_interp = np.empty((num_particle, num_bunch), dtype=dtype)

            if not cugeneralwake:
                device_wt_avg = cuda.device_array((2*self.num_bin_interp-1, num_bunch), dtype=dtype_gpu, stream=stream)
            
            if cugeneralwake:
                device_wtx_avg = cuda.device_array((2*self.num_bin_interp-1, num_bunch), dtype=dtype_gpu, stream=stream)
                device_wty_avg = cuda.device_array((2*self.num_bin_interp-1, num_bunch), dtype=dtype_gpu, stream=stream)

                device_wl_avg_upper = cuda.device_array((2*self.num_bin_interp-1, num_bunch), dtype=dtype_gpu, stream=stream)
                device_wtx_avg_upper = cuda.device_array((2*self.num_bin_interp-1, num_bunch), dtype=dtype_gpu, stream=stream)
                device_wty_avg_upper = cuda.device_array((2*self.num_bin_interp-1, num_bunch), dtype=dtype_gpu, stream=stream)

            device_wp_x = cuda.device_array((self.num_bin_interp, num_bunch), dtype=dtype_gpu, stream=stream)
            device_wp_y = cuda.device_array((self.num_bin_interp, num_bunch), dtype=dtype_gpu, stream=stream)
            device_wp_tau = cuda.device_array((self.num_bin_interp, num_bunch), dtype=dtype_gpu, stream=stream)

            device_wp_x_interp = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)
            device_wp_y_interp = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)
            device_wp_tau_interp = cuda.device_array((num_particle, num_bunch), dtype=dtype_gpu, stream=stream)

            device_density_profile = cuda.device_array((self.num_bin, num_bunch), dtype=dtype_gpu, stream=stream)
            device_density_profile_interp = cuda.device_array((self.num_bin_interp, num_bunch), dtype=dtype_gpu, stream=stream)
            device_profile = cuda.device_array((self.num_bin, num_bunch), dtype=dtype_gpu, stream=stream)
            device_sum_bin_x = cuda.device_array((self.num_bin, num_bunch), dtype=dtype_gpu, stream=stream)
            device_sum_bin_y = cuda.device_array((self.num_bin, num_bunch), dtype=dtype_gpu, stream=stream)
            device_dip_x = cuda.device_array((self.num_bin, num_bunch), dtype=dtype_gpu, stream=stream)
            device_dip_y = cuda.device_array((self.num_bin, num_bunch), dtype=dtype_gpu, stream=stream)
            device_dip_x_interp = cuda.device_array((self.num_bin_interp, num_bunch), dtype=dtype_gpu, stream=stream)
            device_dip_y_interp = cuda.device_array((self.num_bin_interp, num_bunch), dtype=dtype_gpu, stream=stream)

            device_prefix_sum_x_squared = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_xp_squared = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_x_xp = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_y_squared = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_yp_squared = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_y_yp = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_tau_squared = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_delta_squared = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            # device_prefix_sum_tau_delta = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_tau = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_delta = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_x = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_y = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
                
            device_prefix_sum_x_lrrw = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_y_lrrw = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)
            device_prefix_sum_tau_lrrw = cuda.device_array((num_particle_red, num_bunch), dtype=dtype_gpu, stream=stream)

            device_axis_sum_x_squared = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_xp_squared = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_x_xp = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_y_squared = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_yp_squared = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_y_yp = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_tau_squared = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_delta_squared = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            # device_axis_sum_tau_delta = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_tau = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_delta = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_x = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_y = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)

            device_axis_sum_x_lrrw = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_y_lrrw = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_axis_sum_tau_lrrw = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)

            device_sum_kick_x = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_sum_kick_y = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
            device_sum_kick_tau = cuda.device_array((num_bunch,), dtype=dtype_gpu, stream=stream)
                
            for k in range(turns):
                if culm:
                    longmap1_kernel[blockpergrid, threadperblock, stream](
                            Bij, num_particle, num_bunch, device_delta, self.ring.U0, self.ring.E0
                        )
                    
                    longmap2_kernel[blockpergrid, threadperblock, stream](
                            Bij, num_particle, num_bunch, device_tau, device_delta, self.ring.ac, self.ring.T0
                        )

                if cusr:
                    if dblPrec6D :
                        rng_kernel64[blockpergrid, threadperblock, stream](
                                Bij, num_particle, num_bunch, device_xp, device_yp, device_delta, device_xp_sr, device_yp_sr,
                                device_delta_sr, device_rand_xp, device_rand_yp, device_rand_delta, sigma_xp, sigma_yp,
                                self.ring.sigma_delta, self.ring.T0, tau_h, tau_v, tau_l, turns, rng_states1, k
                                )
                        
                    if not dblPrec6D :
                        rng_kernel32[blockpergrid, threadperblock, stream](
                                Bij, num_particle, num_bunch, device_xp, device_yp, device_delta, device_xp_sr, device_yp_sr,
                                device_delta_sr, device_rand_xp, device_rand_yp, device_rand_delta, sigma_xp, sigma_yp,
                                self.ring.sigma_delta, self.ring.T0, tau_h, tau_v, tau_l, turns, rng_states1, k
                                )
                    
                    sr_kernel[blockpergrid, threadperblock, stream](
                            Bij, num_particle, num_bunch, device_rand_xp, device_rand_yp, device_rand_delta, device_xp_sr,
                            device_yp_sr, device_delta_sr, device_xp, device_yp, device_delta
                            )
                
                if cutm:
                    transmap_kernel[blockpergrid, threadperblock, stream](
                            Bij, num_particle, num_bunch, device_x, device_xp, device_y, device_yp, device_delta,
                            device_x_tm, device_xp_tm, device_y_tm, device_yp_tm, dispersion_x, dispersion_xp,
                            dispersion_y, dispersion_yp, tune_x, tune_y, chro_x, chro_y, pi, alpha_x, alpha_y,
                            beta_x, beta_y, gamma_x, gamma_y
                            )
                    
                    if not rectangularaperture:    
                        tm_conversion_kernel[blockpergrid, threadperblock, stream](
                                Bij, num_particle, num_bunch, device_x_tm, device_xp_tm, device_y_tm, device_yp_tm,
                                device_x, device_xp, device_y, device_yp
                                )
                        
                    if rectangularaperture:
                        aperture_tm_conversion_kernel[blockpergrid, threadperblock, stream](
                                Bij, num_particle, num_bunch, device_x_tm, device_xp_tm, device_y_tm, device_yp_tm,
                                device_x, device_xp, device_y, device_yp, device_aperture
                                )
                
                if rectangularaperture:
                    rectangular_aperture_kernel[blockpergrid, threadperblock, stream](
                            Bij, num_particle, num_bunch, device_x, device_y, self.x_aperture_radius, self.y_aperture_radius,
                            device_aperture
                            )
                    
                if curfmc:
                    rfc_kernel[blockpergrid, threadperblock, stream](
                            Bij, num_particle, num_bunch, device_tau, device_delta, self.ring.omega1, self.ring.E0,
                            self.Vc1, self.Vc2, self.m1, self.m2, self.theta1, self.theta2, curfhc
                            )
                    
                if cusbwake:
                    if not rectangularaperture:
                        if dblPrec6D :
                            mm_pr_kernel64[blockpergrid_par, threadperblock, stream](
                                    num_bunch, device_tau, device_prefix_min_tau, device_prefix_max_tau, Jij
                                    )
                            
                        if not dblPrec6D :
                            mm_pr_kernel32[blockpergrid_par, threadperblock, stream](
                                    num_bunch, device_tau, device_prefix_min_tau, device_prefix_max_tau, Jij
                                    )
                
                if not rectangularaperture:
                    if not cugeneralwake:
                        initialize_gm_kernel[blockpergrid_pad, threadperblock, stream](
                                device_prefix_min_tau, device_prefix_max_tau, device_axis_min_tau, device_axis_max_tau,
                                device_axis_sum_x, device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                device_axis_sum_y, device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                device_axis_sum_delta, device_density_profile, device_profile, device_sum_bin_x, device_sum_bin_y,
                                device_wl_avg, device_wt_avg, device_wp_x, device_wp_y, device_wp_tau, device_axis_sum_x_lrrw,
                                device_axis_sum_y_lrrw, device_axis_sum_tau_lrrw, device_sum_kick_x, device_sum_kick_y,
                                device_sum_kick_tau, Bij, num_bunch, self.num_bin, self.num_bin_interp, k
                                )
                
                    if cugeneralwake:
                        general_initialize_gm_kernel[blockpergrid_pad, threadperblock, stream](
                                device_prefix_min_tau, device_prefix_max_tau, device_axis_min_tau, device_axis_max_tau,
                                device_axis_sum_x, device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                device_axis_sum_y, device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                device_axis_sum_delta, device_density_profile, device_profile, device_sum_bin_x,
                                device_sum_bin_y, device_wl_avg, device_wtx_avg, device_wty_avg, device_wp_x, device_wp_y,
                                device_wp_tau, device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, device_axis_sum_tau_lrrw,
                                device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau, Bij, num_bunch, self.num_bin,
                                self.num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper, device_wty_avg_upper, k
                                )
                
                if rectangularaperture:
                    if not cugeneralwake:
                        aperture_initialize_gm_kernel[blockpergrid_pad, threadperblock, stream](
                                inf, device_num_particle_alive, device_axis_min_tau, device_axis_max_tau,
                                device_axis_sum_x, device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                device_axis_sum_y, device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau, device_axis_sum_delta,
                                device_density_profile, device_profile, device_sum_bin_x, device_sum_bin_y, device_wl_avg, device_wt_avg,
                                device_wp_x, device_wp_y, device_wp_tau, device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, device_axis_sum_tau_lrrw,
                                device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau, Bij, num_bunch, self.num_bin, self.num_bin_interp, k
                                )
                        
                    if cugeneralwake:
                        aperture_general_initialize_gm_kernel[blockpergrid_pad, threadperblock, stream](
                                inf, device_num_particle_alive, device_axis_min_tau, device_axis_max_tau,
                                device_axis_sum_x, device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                device_axis_sum_y, device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau, device_axis_sum_delta,
                                device_density_profile, device_profile, device_sum_bin_x, device_sum_bin_y, device_wl_avg, device_wtx_avg,
                                device_wty_avg, device_wp_x, device_wp_y, device_wp_tau, device_axis_sum_x_lrrw, device_axis_sum_y_lrrw,
                                device_axis_sum_tau_lrrw, device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau, Bij,
                                num_bunch, self.num_bin, self.num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper, device_wty_avg_upper,
                                k
                                )

                    aperture_num_particle_alive_kernel[blockpergrid, threadperblock, stream](
                            Bij, num_particle, num_bunch, device_num_particle_alive, device_aperture
                            )

                if cusbwake:
                    if not rectangularaperture:
                        mm_results_kernel[blockpergrid_red, threadperblock, stream](
                                Bij, num_bunch, device_prefix_min_tau, device_prefix_max_tau, device_axis_min_tau, device_axis_max_tau,
                                num_particle_red
                                )
                        
                    if rectangularaperture:
                        aperture_mm_results_kernel[blockpergrid, threadperblock, stream](
                                Bij, num_particle, num_bunch, device_tau, device_axis_min_tau, device_axis_max_tau, device_aperture)

                    binning1_kernel[num_bunch_red, self.thread_per_block, stream](
                            num_bunch, self.num_bin, self.num_bin_interp, device_axis_min_tau, device_axis_max_tau,
                            device_axis_min_tau_interp, device_axis_max_tau_interp, device_half_d_bin_tau,
                            device_half_d_bin_tau_interp, t0, device_norm_lim_interp
                            )
                    
                    binning2_kernel[num_bunch_red, self.thread_per_block, stream](
                            num_bunch, device_axis_min_tau, device_axis_max_tau, device_axis_min_tau_interp,
                            device_axis_max_tau_interp, device_half_d_bin_tau, device_half_d_bin_tau_interp
                            )

                    binning3_kernel[num_bunch_red, self.thread_per_block, stream](
                            num_bunch, self.num_bin, self.num_bin_interp, device_axis_min_tau, device_axis_min_tau_interp,
                            device_bin_tau, device_bin_tau_interp, device_half_d_bin_tau, device_half_d_bin_tau_interp
                            )
                    
                    if not rectangularaperture:
                        sorting_kernel[blockpergrid, threadperblock, stream](
                                Bij, num_particle, num_bunch, self.num_bin, device_tau, device_half_d_bin_tau, device_bin_tau,
                                device_density_profile, device_profile, device_x, device_y, device_sum_bin_x, device_sum_bin_y
                                )
                        
                    if rectangularaperture:
                        aperture_sorting_kernel[blockpergrid, threadperblock, stream](
                                Bij, num_particle, num_bunch, self.num_bin, device_tau, device_half_d_bin_tau, device_bin_tau,
                                device_density_profile, device_profile, device_x, device_y, device_sum_bin_x, device_sum_bin_y,
                                device_num_particle_alive, device_aperture
                                )
                    
                    dipole_moment_kernel[blockpergrid_bin, threadperblock, stream](
                            Bij, num_bunch, self.num_bin, device_profile, device_sum_bin_x, device_sum_bin_y, device_dip_x,
                            device_dip_y
                            )

                    nan_to_zero_kernel[blockpergrid_bin, threadperblock, stream](
                            Bij, num_bunch, self.num_bin, device_profile, device_dip_x, device_dip_y
                            )
                    
                    density_profile_interp_kernel[blockpergrid_bin_interp, threadperblock, stream](
                            Bij, num_bunch, self.num_bin, self.num_bin_interp, device_bin_tau, device_bin_tau_interp,
                            device_density_profile, device_density_profile_interp, device_dip_x, device_dip_y,
                            device_dip_x_interp, device_dip_y_interp
                            )
                    
                    if cugeneralwake:
                        if not cugeneralwakeshortlong:
                            general_wake1_kernel[blockpergrid_wake, threadperblock, stream](
                                    Bij, num_bunch, num_wake_function, self.num_bin_interp, device_wl_avg_upper,
                                    device_wtx_avg_upper, device_wty_avg_upper, device_bin_tau_interp,
                                    device_axis_min_tau_interp, device_wake_function_time, wake_function_time_interval,
                                    device_wake_function_integ_wl, device_wake_function_integ_wtx,
                                    device_wake_function_integ_wty
                                    )
                        
                            general_wake2_kernel[blockpergrid_pad, threadperblock, stream](
                                    Bij, num_bunch, self.num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper,
                                    device_wty_avg_upper, device_wl_avg, device_wtx_avg, device_wty_avg,
                                    device_half_d_bin_tau_interp
                                    )
                        
                        if cugeneralwakeshortlong:
                            general_wake1_short_long_kernel[blockpergrid_wake, threadperblock, stream](
                                    Bij, num_bunch, num_wake_function, self.num_bin_interp, device_wl_avg_upper,
                                    device_wtx_avg_upper, device_wty_avg_upper, device_bin_tau_interp,
                                    device_axis_min_tau_interp, device_wake_function_time, wake_function_time_interval,
                                    device_wake_function_integ_wl, device_wake_function_integ_wtx,
                                    device_wake_function_integ_wty, self.t_crit, amp_wl_long_integ, amp_wt_long_integ,
                                    device_half_d_bin_tau_interp, yoklong, yokxdip, yokydip
                                    )
                            
                            general_wake2_short_long_kernel[blockpergrid_pad, threadperblock, stream](
                                    Bij, num_bunch, self.num_bin_interp, device_wl_avg_upper, device_wtx_avg_upper,
                                    device_wty_avg_upper, device_wl_avg, device_wtx_avg, device_wty_avg, self.t_crit,
                                    device_bin_tau_interp, device_axis_min_tau_interp
                                    )

                        general_wake_convolution_kernel[blockpergrid_pad, threadperblock, stream](
                                Bij, num_bunch, self.num_bin_interp, device_wl_avg, device_wtx_avg, device_wty_avg, device_wp_x,
                                device_wp_y, device_wp_tau, device_density_profile_interp, device_dip_x_interp,
                                device_dip_y_interp, device_half_d_bin_tau_interp
                                )
                    
                    if not cugeneralwake:
                        if not curwcircularseries:
                            circular_rw_wake_kernel[num_bunch_red, self.thread_per_block, stream](
                                    num_bunch, self.num_bin_interp, amp_wl_integ, amp_wt_integ, pi, device_norm_lim_interp,
                                    device_half_d_bin_tau_interp, device_wl_avg, device_wt_avg
                                    )
                        
                        if curwcircularseries:
                            series_circular_rw_wake_kernel[num_bunch_red, self.thread_per_block, stream](
                                    num_bunch, self.num_bin_interp, t0, device_half_d_bin_tau_interp, amp_common, amp_wl_25_integ,
                                    amp_wl_long_integ, amp_wt_24_integ, amp_wt_long_integ, device_norm_lim_interp, device_wl_avg,
                                    device_wt_avg
                                    )
                    
                        wake_convolution_kernel[blockpergrid_pad, threadperblock, stream](
                                Bij, num_bunch, self.num_bin_interp, device_wl_avg, device_wt_avg, device_wp_x, device_wp_y,
                                device_wp_tau, device_density_profile_interp, device_dip_x_interp, device_dip_y_interp,
                                device_half_d_bin_tau_interp
                                )
                    
                    wake_interp_kernel[blockpergrid, threadperblock, stream](
                            Bij, num_particle, num_bunch, self.num_bin_interp, device_wp_x, device_wp_y, device_wp_tau,
                            device_bin_tau_interp, device_tau, device_wp_x_interp, device_wp_y_interp, device_wp_tau_interp
                            )
                    
                    if not rectangularaperture:
                        kick_sb_kernel[blockpergrid, threadperblock, stream](
                                self.ring.E0, Bij, num_particle, num_bunch, device_charge_per_bunch, device_wp_x_interp, device_wp_y_interp,
                                device_wp_tau_interp, device_xp, device_yp, device_delta, self.norm_beta
                                )
                
                    if rectangularaperture:
                        aperture_kick_sb_kernel[blockpergrid, threadperblock, stream](
                                self.ring.E0, Bij, num_particle, num_bunch, device_charge_per_mp, device_num_particle_alive,
                                device_wp_x_interp, device_wp_y_interp, device_wp_tau_interp, device_xp, device_yp, device_delta,
                                self.norm_beta, device_aperture
                                )

                if culrrw:
                    if not rectangularaperture:
                        shift_tables_kernel[blockpergrid_lrrw, threadperblock, stream](
                                num_bunch, turns_lrrw, self.ring.T0, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                device_tau_lrrw_roll, device_x_lrrw_roll, device_y_lrrw_roll
                                )
                    
                        update_tables_kernel[blockpergrid_lrrw, threadperblock, stream](
                                num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                device_tau_lrrw_roll, device_x_lrrw_roll, device_y_lrrw_roll
                                )
                        
                        if dblPrec6D:
                            mean_ps_kernel64[blockpergrid, threadperblock, stream](
                                    device_tau, device_x, device_y, device_prefix_sum_tau_lrrw, device_prefix_sum_x_lrrw,
                                    device_prefix_sum_y_lrrw, num_bunch, Jij
                                    )
                                                
                        if not dblPrec6D:
                            mean_ps_kernel32[blockpergrid, threadperblock, stream](
                                    device_tau, device_x, device_y, device_prefix_sum_tau_lrrw, device_prefix_sum_x_lrrw,
                                    device_prefix_sum_y_lrrw, num_bunch, Jij
                                    )
                    
                    if rectangularaperture:
                        aperture_shift_tables_kernel[blockpergrid_lrrw, threadperblock, stream](
                                num_bunch, turns_lrrw, self.ring.T0, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                device_tau_lrrw_roll, device_x_lrrw_roll, device_y_lrrw_roll, device_charge_lrrw,
                                device_charge_lrrw_roll
                                )
                        
                        aperture_update_tables_kernel[blockpergrid_lrrw, threadperblock, stream](
                                num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw, device_tau_lrrw_roll,
                                device_x_lrrw_roll, device_y_lrrw_roll, device_charge_lrrw, device_charge_lrrw_roll
                                )
                    
                    if not rectangularaperture:
                        mean_as_kernel[blockpergrid_red, threadperblock, stream](
                                device_prefix_sum_tau_lrrw, device_prefix_sum_x_lrrw, device_prefix_sum_y_lrrw,
                                device_axis_sum_tau_lrrw, device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, Bij,
                                num_bunch, num_particle_red
                                )
                        
                        mean_tables_kernel[num_bunch_red, self.thread_per_block, stream](
                                device_init_long_pos, num_bunch, num_particle, device_axis_sum_tau_lrrw, device_axis_sum_x_lrrw,
                                device_axis_sum_y_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw
                                )
                    
                        get_kick_btb_kernel[blockpergrid_lrrw, threadperblock, stream](
                                num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw, device_sum_kick_tau,
                                device_sum_kick_x, device_sum_kick_y, device_charge_per_bunch, amp_wl_long, amp_wt_long,
                                yoklong, yokxdip, yokydip
                                    )
                    
                    if rectangularaperture:
                        aperture_mean_as_kernel[blockpergrid, threadperblock, stream](
                                device_tau, device_x, device_y, device_axis_sum_tau_lrrw, device_axis_sum_x_lrrw,
                                device_axis_sum_y_lrrw, Bij, num_particle, num_bunch, device_aperture
                                )
                        
                        aperture_mean_tables_kernel[num_bunch_red, self.thread_per_block, stream](
                                device_init_long_pos, device_charge_per_mp, num_bunch, device_num_particle_alive, device_axis_sum_tau_lrrw,
                                device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                device_charge_lrrw
                                )
                        
                        aperture_get_kick_btb_kernel[blockpergrid_lrrw, threadperblock, stream](
                                num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw, device_sum_kick_tau,
                                device_sum_kick_x, device_sum_kick_y, device_charge_lrrw, amp_wl_long, amp_wt_long,
                                yoklong, yokxdip, yokydip
                                )
                    
                    kick_btb_kernel[blockpergrid, threadperblock, stream](
                            Bij, num_particle, num_bunch, device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau, device_xp,
                            device_yp, device_delta, self.ring.E0, self.norm_beta
                            )
                    
                if not rectangularaperture: 
                    if curm:
                        if (k + 1) % curm_ti == 0:
                            if dblPrec6D:
                                monitor_ps_kernel64[blockpergrid, threadperblock, stream](
                                        device_x, device_xp, device_y, device_yp, device_tau, device_delta, device_prefix_sum_x_squared,
                                        device_prefix_sum_xp_squared, device_prefix_sum_x_xp, device_prefix_sum_y_squared,
                                        device_prefix_sum_yp_squared, device_prefix_sum_y_yp, device_prefix_sum_tau_squared,
                                        device_prefix_sum_delta_squared, device_prefix_sum_tau, device_prefix_sum_delta, device_prefix_sum_x,
                                        device_prefix_sum_y, Bij, num_bunch
                                        )
                                
                            if not dblPrec6D:
                                monitor_ps_kernel32[blockpergrid, threadperblock, stream](
                                        device_x, device_xp, device_y, device_yp, device_tau, device_delta, device_prefix_sum_x_squared,
                                        device_prefix_sum_xp_squared, device_prefix_sum_x_xp, device_prefix_sum_y_squared,
                                        device_prefix_sum_yp_squared, device_prefix_sum_y_yp, device_prefix_sum_tau_squared,
                                        device_prefix_sum_delta_squared, device_prefix_sum_tau, device_prefix_sum_delta, device_prefix_sum_x,
                                        device_prefix_sum_y, Bij, num_bunch
                                        )
                        
                            monitor_as_kernel[blockpergrid_red, threadperblock, stream](
                                    device_prefix_sum_x_squared, device_prefix_sum_xp_squared, device_prefix_sum_x_xp,
                                    device_prefix_sum_y_squared, device_prefix_sum_yp_squared, device_prefix_sum_y_yp,
                                    device_prefix_sum_tau_squared, device_prefix_sum_delta_squared, device_prefix_sum_tau,
                                    device_prefix_sum_delta, device_prefix_sum_x, device_prefix_sum_y, device_axis_sum_x_squared,
                                    device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared,
                                    device_axis_sum_yp_squared, device_axis_sum_y_yp, device_axis_sum_tau_squared,
                                    device_axis_sum_delta_squared, device_axis_sum_tau, device_axis_sum_delta, device_axis_sum_x,
                                    device_axis_sum_y, Bij, num_bunch, num_particle_red
                                    )

                            monitor_results_kernel[num_bunch_red, self.thread_per_block, stream](
                                    device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                    device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                    device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                    device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, device_bunch_length,
                                    device_energy_spread, device_Jx, device_Jy, device_beam_sizeX, device_beam_sizeY,
                                    device_beam_comS, device_beam_comX, device_beam_comY, alpha_x, alpha_y, beta_x, beta_y,
                                    gamma_x, gamma_y, num_bunch, num_particle, k
                                    )
                    
                    if not curm:
                        if dblPrec6D:
                            monitor_ps_kernel64[blockpergrid_par, threadperblock, stream](
                                    device_x, device_xp, device_y, device_yp, device_tau, device_delta, device_prefix_sum_x_squared,
                                    device_prefix_sum_xp_squared, device_prefix_sum_x_xp, device_prefix_sum_y_squared,
                                    device_prefix_sum_yp_squared, device_prefix_sum_y_yp, device_prefix_sum_tau_squared,
                                    device_prefix_sum_delta_squared, device_prefix_sum_tau, device_prefix_sum_delta, device_prefix_sum_x,
                                    device_prefix_sum_y, num_bunch, Jij
                                    )
                            
                        if not dblPrec6D:
                            monitor_ps_kernel32[blockpergrid_par, threadperblock, stream](
                                    device_x, device_xp, device_y, device_yp, device_tau, device_delta, device_prefix_sum_x_squared,
                                    device_prefix_sum_xp_squared, device_prefix_sum_x_xp, device_prefix_sum_y_squared,
                                    device_prefix_sum_yp_squared, device_prefix_sum_y_yp, device_prefix_sum_tau_squared,
                                    device_prefix_sum_delta_squared, device_prefix_sum_tau, device_prefix_sum_delta, device_prefix_sum_x,
                                    device_prefix_sum_y, num_bunch, Jij
                                    )
                        
                        monitor_as_kernel[blockpergrid_red, threadperblock, stream](
                                device_prefix_sum_x_squared, device_prefix_sum_xp_squared, device_prefix_sum_x_xp,
                                device_prefix_sum_y_squared, device_prefix_sum_yp_squared, device_prefix_sum_y_yp,
                                device_prefix_sum_tau_squared, device_prefix_sum_delta_squared, device_prefix_sum_tau,
                                device_prefix_sum_delta, device_prefix_sum_x, device_prefix_sum_y, device_axis_sum_x_squared,
                                device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared, device_axis_sum_yp_squared,
                                device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, Bij, num_bunch, num_particle_red
                                )


                        monitor_results_kernel[num_bunch_red, self.thread_per_block, stream](
                                device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared,
                                device_axis_sum_yp_squared, device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared,
                                device_axis_sum_tau, device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, device_bunch_length,
                                device_energy_spread, device_Jx, device_Jy, device_beam_sizeX, device_beam_sizeY, device_beam_comS,
                                device_beam_comX, device_beam_comY, alpha_x, alpha_y, beta_x, beta_y, gamma_x, gamma_y, num_bunch,
                                num_particle, k
                                )
                        
                if rectangularaperture:
                    if curm:
                        if (k + 1) % curm_ti == 0:
                            aperture_monitor_as_kernel[blockpergrid, threadperblock, stream](
                                    device_x, device_xp, device_y, device_yp, device_tau, device_delta, device_axis_sum_x_squared,
                                    device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared, 
                                    device_axis_sum_yp_squared, device_axis_sum_y_yp, device_axis_sum_tau_squared,
                                    device_axis_sum_delta_squared, device_axis_sum_tau, device_axis_sum_delta,
                                    device_axis_sum_x, device_axis_sum_y, Bij, num_particle, num_bunch, device_aperture
                                    )
                        
                            aperture_monitor_results_kernel[num_bunch_red, self.thread_per_block, stream](
                                    device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                    device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                    device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                    device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, device_bunch_length,
                                    device_energy_spread, device_Jx, device_Jy, device_beam_sizeX, device_beam_sizeY,
                                    device_beam_comS, device_beam_comX, device_beam_comY, alpha_x, alpha_y, beta_x,
                                    beta_y, gamma_x, gamma_y, num_bunch, device_num_particle_alive, k
                                    )
                    
                    if not curm:
                        aperture_monitor_as_kernel[blockpergrid, threadperblock, stream](
                                device_x, device_xp, device_y, device_yp, device_tau, device_delta, device_axis_sum_x_squared,
                                device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared, device_axis_sum_yp_squared,
                                device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau,
                                device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, Bij, num_bunch, device_aperture
                                )
                      
                        aperture_monitor_results_kernel[num_bunch_red, self.thread_per_block, stream](
                                device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared,
                                device_axis_sum_yp_squared, device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared,
                                device_axis_sum_tau, device_axis_sum_delta, device_axis_sum_x, device_axis_sum_y, device_bunch_length,
                                device_energy_spread, device_Jx, device_Jy, device_beam_sizeX, device_beam_sizeY, device_beam_comS,
                                device_beam_comX, device_beam_comY, alpha_x, alpha_y, beta_x, beta_y, gamma_x, gamma_y, num_bunch,
                                num_particle, device_num_particle_alive, k
                                )
                if monitordip:
                    monitor_dipole_moments_kernel[blockpergrid_bin, threadperblock, stream](
                            device_dip_x, device_dip_y, device_dipX, device_dipY, Bij, num_bunch, self.num_bin, k
                            )
                
            # device_x.copy_to_host(x_single, stream=stream)
            # device_xp.copy_to_host(xp_single, stream=stream)
            # device_y.copy_to_host(y_single, stream=stream)
            # device_yp.copy_to_host(yp_single, stream=stream)
            device_tau.copy_to_host(tau, stream=stream)
            # device_delta.copy_to_host(delta_single, stream=stream)

            # device_beam_emitX.copy_to_host(beam_emitX, stream=stream)
            # device_beam_emitY.copy_to_host(beam_emitY, stream=stream)
            # device_beam_emitS.copy_to_host(beam_emitS, stream=stream)

            device_bunch_length.copy_to_host(bunch_length, stream=stream)
            device_energy_spread.copy_to_host(energy_spread, stream=stream)
            device_Jx.copy_to_host(Jx, stream=stream)
            device_Jy.copy_to_host(Jy, stream=stream)
            device_beam_sizeX.copy_to_host(beam_sizeX, stream=stream)
            device_beam_sizeY.copy_to_host(beam_sizeY, stream=stream)
            device_beam_comS.copy_to_host(beam_comS, stream=stream)
            device_beam_comX.copy_to_host(beam_comX, stream=stream)
            device_beam_comY.copy_to_host(beam_comY, stream=stream)
            
            if monitordip:
                device_dipX.copy_to_host(dipX, stream=stream)
                device_dipY.copy_to_host(dipY, stream=stream)

            stream.synchronize()

            current_file_path = os.path.dirname(os.path.abspath(__file__))
            data_folder_path = os.path.join(current_file_path, "..", "..", "data")
            os.chdir(data_folder_path)

            for i in range(num_bunch):
                tau_save = tau[:, i]
                filename = f"gpu_data_{int(self.idxs*self.ring.h)}ma_k4gsr_chro0_{int(num_bunch)}_{i}.hdf5"
                with hp.File(filename, "w") as f:
                    f.create_dataset("bunch_length", data=bunch_length[:, i])
                    f.create_dataset("energy_spread", data=energy_spread[:, i])
                    f.create_dataset("Jx", data=Jx[:, i])
                    f.create_dataset("Jy", data=Jy[:, i])
                    f.create_dataset("tau_save", data=tau_save)
                    f.create_dataset("beam_sizeX", data=beam_sizeX[:, i])
                    f.create_dataset("beam_sizeY", data=beam_sizeY[:, i])
                    f.create_dataset("beam_comS", data=beam_comS[:, i])
                    f.create_dataset("beam_comX", data=beam_comX[:, i])
                    f.create_dataset("beam_comY", data=beam_comY[:, i])
                    if monitordip:
                        f.create_dataset("dipX", data=dipX[:, :, i])
                        f.create_dataset("dipY", data=dipY[:, :, i])

        else:
            raise ValueError("To perform GPU calculations, CUDA_PARALLEL must be enabled in the mybeam.init_beam.")
