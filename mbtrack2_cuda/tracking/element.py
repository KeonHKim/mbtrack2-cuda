# -*- coding: utf-8 -*-
"""
This module defines the most basic elements for tracking, including Element,
an abstract base class which is to be used as mother class to every elements
included in the tracking.
"""
import numpy as np
import time
import numba
import os
import pickle
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
from math import sqrt, sin, cos, isnan
from scipy.constants import mu_0, c, pi
from abc import ABCMeta, abstractmethod
from functools import wraps
from copy import deepcopy
from mbtrack2_cuda.tracking.particles import Beam
import matplotlib.pyplot as plt

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
    def __init__(self, ring, m, Vc, theta, n_bin, rho, radius, length):
        self.ring = ring
        self.m = m
        self.Vc = Vc
        self.theta = theta
        self.alpha = self.ring.optics.local_alpha
        self.beta = self.ring.optics.local_beta
        self.gamma = self.ring.optics.local_gamma
        self.dispersion = self.ring.optics.local_dispersion
        self.n_bin = n_bin
        self.rho = rho
        self.radius = radius
        self.length = length
        if self.ring.adts is not None:
            self.adts_poly = [np.poly1d(self.ring.adts[0]),
                              np.poly1d(self.ring.adts[1]),
                              np.poly1d(self.ring.adts[2]), 
                              np.poly1d(self.ring.adts[3])]
            
    def track(self, bunch, turns, turns_lrrw, culm, cutm, cusr, curfc, curw, cubm):
        """
        Tracking method for the element

        """
        # No Selection Error
        if (culm == False) and (cutm == False) and (cusr == False) and (curfc == False):
            raise ValueError("There is nothing to track.")
        
        @cuda.jit(device=True)
        def wl_25_integ(amp_common, amp_wl_25_integ, norm_t):
            """
            Integrated series expanded short-range longitudinal wake function up to 25th order

            """

            return ( amp_wl_25_integ * ( norm_t + (1.074569931823542*norm_t)**4/4 + (0.7076774448131904*norm_t)**7/7
                    + (0.5187411408284097*norm_t)**10/10 + (0.4075262743459792*norm_t)**13/13 + (0.3349148429183678*norm_t)**16/16
                    + (0.2839707617267887*norm_t)**19/19 + (0.24632755711646917*norm_t)**22/22 + (0.21741052713420603*norm_t)**25/25
                    + (0.19451724523154354*norm_t)**28/28 + (0.1759522529668746*norm_t)**31/31 + (0.16059951902667116*norm_t)**34/34
                    + (0.1476950951140299*norm_t)**37/37 + (0.1366987737767119*norm_t)**40/40 + (0.12721795478105152*norm_t)**43/43
                    + (0.11896055968293133*norm_t)**46/46 + (0.11170485812348441*norm_t)**49/49 + (0.10527952992003195*norm_t)**52/52
                    + (0.09955013421701707*norm_t)**55/55 + (0.09440970876302733*norm_t)**58/58 + (0.08977210146369936*norm_t)**61/61
                    + (0.08556715156561916*norm_t)**64/64 + (0.08173714900619293*norm_t)**67/67 + (0.07823419353640372*norm_t)**70/70
                    + (0.07501819794305475*norm_t)**73/73 + (0.07205535941962744*norm_t)**76/76
                    - amp_common * ( (2.577576059909017*norm_t)**(2.5)/5 + (1.1509922245183923*norm_t)**(5.5)/11
                    + (0.7245636912116582*norm_t)**(8.5)/17 + (0.5253761406648592*norm_t)**(11.5)/23 + (0.41099027060230875*norm_t)**(14.5)/29
                    + (0.33704170196010497*norm_t)**(17.5)/35 + (0.28541612243560294*norm_t)**(20.5)/41 + (0.24737894998121937*norm_t)**(23.5)/47
                    + (0.21821307320741443*norm_t)**(26.5)/53 + (0.19515204813901926*norm_t)**(29.5)/59 + (0.17646825768725524*norm_t)**(32.5)/65
                    + (0.16102807823666604*norm_t)**(35.5)/71 + (0.14805726797068675*norm_t)**(38.5)/77 + (0.137009260651878*norm_t)**(41.5)/83
                    + (0.12748734882778276*norm_t)**(44.5)/89 + (0.11919670060047279*norm_t)**(47.5)/95 + (0.11191368072380277*norm_t)**(50.5)/101
                    + (0.1054656152340559*norm_t)**(53.5)/107 + (0.09971707851632573*norm_t)**(56.5)/113 + (0.09456037730712599*norm_t)**(59.5)/119
                    + (0.08990880709153554*norm_t)**(62.5)/125 + (0.0856917830023976*norm_t)**(65.5)/131 + (0.08185126474468153*norm_t)**(68.5)/137
                    + (0.07833909166227981*norm_t)**(71.5)/143 + (0.07511496883839254*norm_t)**(74.5)/149 ) ) )
        
        @cuda.jit(device=True, inline=True)
        def wl_long_integ(amp_wl_long_integ, t):
            """
            Integrated long-range longitudinal wake function

            """

            return ( amp_wl_long_integ / sqrt(t) )
        
        @cuda.jit(device=True)
        def wt_24_integ(amp_common, amp_wt_24_integ, norm_t):
            """
            Integrated series expanded short-range transeverse wake function up to 24th order

            """

            return ( amp_wt_24_integ * ( (norm_t)**2 + (0.9221079114817278*norm_t)**5/5 + (0.6318259065240571*norm_t)**8/8
                    + (0.47568728010371913*norm_t)**11/11 + (0.3801337254877733*norm_t)**14/14 + (0.3160505063118279*norm_t)**17/17
                    + (0.2702239044700352*norm_t)**20/20 + (0.23587994268147205*norm_t)**23/23 + (0.20920927345893964*norm_t)**26/26
                    + (0.1879122704184599*norm_t)**29/29 + (0.17052119913691632*norm_t)**32/32 + (0.15605639798554585*norm_t)**35/35
                    + (0.14383952358668206*norm_t)**38/38 + (0.13338625879363325*norm_t)**41/41 + (0.12434172544096447*norm_t)**44/44
                    + (0.1164400391314344*norm_t)**47/47 + (0.10947811179627656*norm_t)**50/50 + (0.10329817947523212*norm_t)**53/53
                    + (0.09777584776432816*norm_t)**56/56 + (0.09281172620504952*norm_t)**59/59 + (0.08832545593197942*norm_t)**62/62
                    + (0.08425136901778911*norm_t)**65/65 + (0.08053528260900018*norm_t)**68/68 + (0.07713209652508592*norm_t)**71/71
                    + (0.0740039690318997*norm_t)**74/74
                    - amp_common * ( (1.8451595158925989*norm_t)**(3.5)/7 + (0.9640221048331283*norm_t)**(6.5)/13 
                    + (0.6436638912723234*norm_t)**(9.5)/19 + (0.48090499800039915*norm_t)**(12.5)/25 + (0.3830363208137494*norm_t)**(15.5)/31
                    + (0.3179031996606376*norm_t)**(18.5)/37 + (0.2715155271645874*norm_t)**(21.5)/43 + (0.23683638921378813*norm_t)**(24.5)/49
                    + (0.20994890555758866*norm_t)**(27.5)/55 + (0.1885031096884124*norm_t)**(30.5)/61 + (0.17100518766778883*norm_t)**(33.5)/67
                    + (0.15646086265771714*norm_t)**(36.5)/73 + (0.14418307266691813*norm_t)**(39.5)/79 + (0.13368202884097732*norm_t)**(42.5)/85
                    + (0.12459927206340426*norm_t)**(45.5)/91 + (0.1166664905981537*norm_t)**(48.5)/97 + (0.10967890119726531*norm_t)**(51.5)/103
                    + (0.10347752533150813*norm_t)**(54.5)/109 + (0.09793707883805237*norm_t)**(57.5)/115 + (0.09295750666092513*norm_t)**(60.5)/121
                    + (0.08845794494378313*norm_t)**(63.5)/127 + (0.08437233616364609*norm_t)**(66.5)/133 + (0.08064619278392919*norm_t)**(69.5)/139
                    + (0.0772341734171156*norm_t)**(72.5)/145 ) ) )
        
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

            return ( amp_wl_long / t**1.5)
        
        @cuda.jit(device=True, inline=True)
        def wt_long(amp_wt_long, t):
            """
            Long-range Transverse wake function

            """

            return ( amp_wt_long / sqrt(t) )
        
        @cuda.jit
        def rng_kernel(num_particle, num_bunch, rng_states,
                       device_rand_xp, device_rand_yp, device_rand_delta):
            """
            Random number generation for synchrotron radiation and radiation damping

            """
            i, j = cuda.grid(2)

            if j < num_particle and i < num_bunch:
                    device_rand_xp[j, i] = xoroshiro128p_normal_float32(rng_states, j + num_particle * i)
                    device_rand_yp[j, i] = xoroshiro128p_normal_float32(rng_states, j + num_particle * i)
                    device_rand_delta[j, i] = xoroshiro128p_normal_float32(rng_states, j + num_particle * i)
                    
        @cuda.jit
        def map_kernel(device_x, device_xp, device_y, device_yp, device_tau, device_delta,
                       U0, E0, ac, T0, sigma_delta, omega1, sigma_xp, sigma_yp,
                       tau_h, tau_v, tau_l, dispersion_x, dispersion_xp, dispersion_y, dispersion_yp,
                       m, Vc, theta, tune_x, tune_y, chro_x, chro_y, pi, alpha_x, alpha_y, beta_x, beta_y,
                       gamma_x, gamma_y, device_rand_xp, device_rand_yp, device_rand_delta,
                       culm, cutm, cusr, curfc, curw, cubm):
            """
            Longitudinal map + transverse map + synchrotron radiation + RF cavity +
            parallel reduction for resistive wall and beam monitor

            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y
            
            tau_shared = cuda.shared.array(threadperblock, numba.float32)
            delta_shared = cuda.shared.array(threadperblock, numba.float32)

            x_shared = cuda.shared.array(threadperblock, numba.float32)
            xp_shared = cuda.shared.array(threadperblock, numba.float32)
            y_shared = cuda.shared.array(threadperblock, numba.float32)
            yp_shared = cuda.shared.array(threadperblock, numba.float32)

            x_shared_f = cuda.shared.array(threadperblock, numba.float32)
            xp_shared_f = cuda.shared.array(threadperblock, numba.float32)
            y_shared_f = cuda.shared.array(threadperblock, numba.float32)
            yp_shared_f = cuda.shared.array(threadperblock, numba.float32)

            x_shared[local_j, local_i] = device_x[j, i]
            xp_shared[local_j, local_i] = device_xp[j, i]
            y_shared[local_j, local_i] = device_y[j, i]
            yp_shared[local_j, local_i] = device_yp[j, i]
            tau_shared[local_j, local_i] = device_tau[j, i]
            delta_shared[local_j, local_i] = device_delta[j, i]
            
            cuda.syncthreads()
  
            # Longitudinal Map
            if culm:
                   
               delta_shared[local_j, local_i] -= U0 / E0
               cuda.syncthreads()

               tau_shared[local_j, local_i] += ac * T0 * delta_shared[local_j, local_i]
               cuda.syncthreads()

            # Transverse Map
            # adts effects are ignored. (Future work)
            if cutm:
                   
                x_shared_f[local_j, local_i] = ( ( cos(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) +
                                                alpha_x * sin(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) )
                                                * x_shared[local_j, local_i] + ( beta_x * sin(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) )
                                                * xp_shared[local_j, local_i] + dispersion_x * delta_shared[local_j, local_i] )

                xp_shared_f[local_j, local_i] = ( ( -1 * gamma_x * sin(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) )
                                                * x_shared[local_j, local_i] + ( cos(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) -
                                                alpha_x * sin(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) ) * xp_shared[local_j, local_i] +
                                                dispersion_xp * delta_shared[local_j, local_i] )

                y_shared_f[local_j, local_i] = ( ( cos(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) +
                                                alpha_y * sin(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) )
                                                * y_shared[local_j, local_i] + ( beta_y * sin(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) )
                                                * yp_shared[local_j, local_i] + dispersion_y * delta_shared[local_j, local_i] )

                yp_shared_f[local_j, local_i] = ( ( -1 * gamma_y * sin(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) )
                                                * y_shared[local_j, local_i] + ( cos(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) -
                                                alpha_y * sin(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) ) * yp_shared[local_j, local_i] +
                                                dispersion_yp * delta_shared[local_j, local_i] )  
                     
                cuda.syncthreads()
                   
                x_shared[local_j, local_i] = x_shared_f[local_j, local_i]
                xp_shared[local_j, local_i] = xp_shared_f[local_j, local_i]
                y_shared[local_j, local_i] = y_shared_f[local_j, local_i]
                yp_shared[local_j, local_i] = yp_shared_f[local_j, local_i]

                cuda.syncthreads()

            # Synchrotron Radiation
            # if cusr:
            #     #rand_xp
            #     xp_shared[local_j, local_i] = ( (1 - 2*T0/tau_h) * xp_shared[local_j, local_i] +
            #         2*sigma_xp*sqrt(T0/tau_h) * device_rand_xp[j, i] )

            #     #rand_yp
            #     yp_shared[local_j, local_i] = ( (1 - 2*T0/tau_v) * yp_shared[local_j, local_i] +
            #         2*sigma_yp*sqrt(T0/tau_v) * device_rand_yp[j, i] )
                          
            #     #rand_delta
            #     delta_shared[local_j, local_i] = ( (1 - 2*T0/tau_l) * delta_shared[local_j, local_i] +
            #         2*sigma_delta*sqrt(T0/tau_l) * device_rand_delta[j, i] )
                          
            #     cuda.syncthreads()

            # RF Cavity
            if curfc:
                delta_shared[local_j, local_i] += Vc / E0 * cos(m * omega1 * tau_shared[local_j, local_i] + theta)
                cuda.syncthreads()

            device_x[j, i] = x_shared[local_j, local_i]
            device_xp[j, i] = xp_shared[local_j, local_i]
            device_y[j, i] = y_shared[local_j, local_i]
            device_yp[j, i] = yp_shared[local_j, local_i]
            device_tau[j, i] = tau_shared[local_j, local_i]
            device_delta[j, i] = delta_shared[local_j, local_i]

        @cuda.jit
        def mm_pr1_kernel(device_tau, device_1st_min_tau, device_1st_max_tau, num_bunch):
            """
            1st parallel reduction of min & max finding for each bunch

            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y

            min_tau_shared = cuda.shared.array(threadperblock, numba.float32)
            max_tau_shared = cuda.shared.array(threadperblock, numba.float32)

            min_tau_shared[local_j, local_i] = device_tau[j, i]
            max_tau_shared[local_j, local_i] = device_tau[j, i]
            cuda.syncthreads()

            s = threadperblock[1]
            s >>= 1
            while s > 0:
                if local_j < s and local_i < num_bunch:
                    min_tau_shared[local_j, local_i] = min(min_tau_shared[local_j, local_i], min_tau_shared[local_j + s, local_i])
                    max_tau_shared[local_j, local_i] = max(max_tau_shared[local_j, local_i], max_tau_shared[local_j + s, local_i])
                cuda.syncthreads()
                s >>= 1
                
            if local_j == 0 and i < num_bunch:
                device_1st_min_tau[cuda.blockIdx.y, i] = min_tau_shared[0, local_i]
                device_1st_max_tau[cuda.blockIdx.y, i] = max_tau_shared[0, local_i]

        @cuda.jit
        def mm_pr2_kernel(device_1st_min_tau, device_1st_max_tau, device_2nd_min_tau, device_2nd_max_tau, num_bunch):
            """
            2nd parallel reduction of min & max finding for each bunch

            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y

            min_tau_shared = cuda.shared.array(threadperblock, numba.float32)
            max_tau_shared = cuda.shared.array(threadperblock, numba.float32)

            min_tau_shared[local_j, local_i] = device_1st_min_tau[j, i]
            max_tau_shared[local_j, local_i] = device_1st_max_tau[j, i]
            cuda.syncthreads()

            s = threadperblock[1]
            s >>= 1
            while s > 0:
                if local_j < s and local_i < num_bunch:
                    min_tau_shared[local_j, local_i] = min(min_tau_shared[local_j, local_i], min_tau_shared[local_j + s, local_i])
                    max_tau_shared[local_j, local_i] = max(max_tau_shared[local_j, local_i], max_tau_shared[local_j + s, local_i])
                cuda.syncthreads()
                s >>= 1
                
            if local_j == 0 and i < num_bunch:
                device_2nd_min_tau[cuda.blockIdx.y, i] = min_tau_shared[0, local_i]
                device_2nd_max_tau[cuda.blockIdx.y, i] = max_tau_shared[0, local_i]
        
        @cuda.jit
        def initialize_gm_kernel(device_2nd_min_tau, device_2nd_max_tau, device_axis_min_tau, device_axis_max_tau,
                                 device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                 device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                 device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau_delta,
                                 device_axis_sum_tau, device_density_profile, device_profile, device_sum_bin_x, device_sum_bin_y,
                                 device_wl_avg, device_wt_avg, device_wp_x, device_wp_y, device_wp_tau,
                                 device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, device_axis_sum_tau_lrrw, device_sum_kick_x,
                                 device_sum_kick_y, device_sum_kick_tau, num_bunch, n_bin, k):
            """
            Initialize global memory arrays

            """
            i, j = cuda.grid(2)

            if k == 0:
                if i < num_bunch:
                    device_axis_min_tau[i] = device_2nd_min_tau[0, i]
                    device_axis_max_tau[i] = device_2nd_max_tau[0, i]
                    device_axis_sum_x_squared[i] = 0
                    device_axis_sum_xp_squared[i] = 0
                    device_axis_sum_x_xp[i] = 0
                    device_axis_sum_y_squared[i] = 0
                    device_axis_sum_yp_squared[i] = 0
                    device_axis_sum_y_yp[i] = 0
                    device_axis_sum_tau_squared[i] = 0
                    device_axis_sum_delta_squared[i] = 0
                    device_axis_sum_tau_delta[i] = 0
                    device_axis_sum_tau[i] = 0
                    device_axis_sum_x_lrrw[i] = 0
                    device_axis_sum_y_lrrw[i] = 0
                    device_axis_sum_tau_lrrw[i] = 0
                    device_sum_kick_x[i] = 0
                    device_sum_kick_y[i] = 0
                    device_sum_kick_tau[i] = 0
                    if j < n_bin:
                        device_profile[j, i] = 0
                        device_density_profile[j, i] = 0
                        device_sum_bin_x[j, i] = 0
                        device_sum_bin_y[j, i] = 0
                        device_wp_x[j, i] = 0
                        device_wp_y[j, i] = 0
                        device_wp_tau[j, i] = 0
                    if (j < 2*n_bin-1):
                        device_wl_avg[j, i] = 0
                        device_wt_avg[j, i] = 0
            
            else:
                if i < num_bunch:
                    device_axis_min_tau[i] = device_2nd_min_tau[0, i]
                    device_axis_max_tau[i] = device_2nd_max_tau[0, i]
                    device_axis_sum_x_squared[i] = 0
                    device_axis_sum_xp_squared[i] = 0
                    device_axis_sum_x_xp[i] = 0
                    device_axis_sum_y_squared[i] = 0
                    device_axis_sum_yp_squared[i] = 0
                    device_axis_sum_y_yp[i] = 0
                    device_axis_sum_tau_squared[i] = 0
                    device_axis_sum_delta_squared[i] = 0
                    device_axis_sum_tau_delta[i] = 0
                    device_axis_sum_tau[i] = 0
                    device_axis_sum_x_lrrw[i] = 0
                    device_axis_sum_y_lrrw[i] = 0
                    device_axis_sum_tau_lrrw[i] = 0
                    device_sum_kick_x[i] = 0
                    device_sum_kick_y[i] = 0
                    device_sum_kick_tau[i] = 0
                    if j < n_bin:
                        device_profile[j, i] = 0
                        device_density_profile[j, i] = 0
                        device_sum_bin_x[j, i] = 0
                        device_sum_bin_y[j, i] = 0
                        device_wp_x[j, i] = 0
                        device_wp_y[j, i] = 0
                        device_wp_tau[j, i] = 0

        @cuda.jit
        def mm_results_kernel(device_2nd_min_tau, device_2nd_max_tau, device_axis_min_tau, device_axis_max_tau, num_bunch, num_red):
            """
            Final min & max values for each bunch

            """
            i = cuda.grid(1)

            if i < num_bunch:
                for idx in range(1, num_red):
                    cuda.atomic.min(device_axis_min_tau, i, device_2nd_min_tau[idx, i])
                    cuda.atomic.max(device_axis_max_tau, i, device_2nd_max_tau[idx, i])

        @cuda.jit
        def binning_kernel(num_bunch, n_bin, device_axis_min_tau, device_axis_max_tau, device_bin_tau, device_half_d_bin_tau,
                           t0, device_norm_lim, device_t):
            """
            Binning kernel for resistive wall instability

            """
            i = cuda.grid(1)
            local_i = cuda.threadIdx.x

            axis_min_tau_shared = cuda.shared.array(threadperblock[0], numba.float32)
            axis_max_tau_shared = cuda.shared.array(threadperblock[0], numba.float32)
            half_d_bin_tau_shared = cuda.shared.array(threadperblock[0], numba.float32)

            axis_min_tau_shared[local_i] = device_axis_min_tau[i]
            axis_max_tau_shared[local_i] = device_axis_max_tau[i]
            cuda.syncthreads()
            
            half_d_bin_tau_shared[local_i] = (axis_max_tau_shared[local_i] - axis_min_tau_shared[local_i]) * 0.5 / (n_bin - 1)
            cuda.syncthreads()

            axis_min_tau_shared[local_i] -= half_d_bin_tau_shared[local_i]
            axis_max_tau_shared[local_i] += half_d_bin_tau_shared[local_i]
            cuda.syncthreads()
            
            device_half_d_bin_tau[i] = half_d_bin_tau_shared[local_i]
            device_norm_lim[i] = half_d_bin_tau_shared[local_i] / t0

            # Center values for each bin
            if i < num_bunch:
                for idx in range(n_bin):
                    device_bin_tau[idx, i] = axis_min_tau_shared[local_i] + half_d_bin_tau_shared[local_i] * (2*idx + 1)
                    device_t[idx, i] = 2*idx*half_d_bin_tau_shared[local_i]

        @cuda.jit
        def sorting_kernel(num_bunch, num_particle, n_bin, device_tau, device_half_d_bin_tau, device_bin_tau,
                           device_density_profile, device_profile, device_x, device_y, device_sum_bin_x,
                           device_sum_bin_y, charge, charge_per_mp):
            """
            Sorting kernel for each bunch & calculation of zero padded charge density profile &
            partial calculation of dipole moments

            """
            i, j = cuda.grid(2)

            if j < num_particle and i < num_bunch:
                for idx in range(n_bin):
                    if ( (device_tau[j, i] >= device_bin_tau[idx, i] - device_half_d_bin_tau[i]) and
                    (device_tau[j, i] < device_bin_tau[idx, i] + device_half_d_bin_tau[i]) ):
                        cuda.atomic.add(device_density_profile, (idx, i), charge_per_mp/(2*device_half_d_bin_tau[i]*charge))
                        cuda.atomic.add(device_profile, (idx, i), 1)
                        cuda.atomic.add(device_sum_bin_x, (idx, i), device_x[j, i])
                        cuda.atomic.add(device_sum_bin_y, (idx, i), device_y[j, i])
        
        @cuda.jit
        def dipole_moment_kernel(num_bunch, n_bin, device_profile, device_density_profile, device_sum_bin_x, device_sum_bin_y,
                                 device_dip_x, device_dip_y, device_profile_dip_x, device_profile_dip_y):
            """
            Calculation of dipole moments

            """
            i, j = cuda.grid(2)

            if i < num_bunch and j < n_bin:
                device_dip_x[j, i] = device_sum_bin_x[j, i] / device_profile[j, i]
                device_dip_y[j, i] = device_sum_bin_y[j, i] / device_profile[j, i]
                device_profile_dip_x[j, i] = device_density_profile[j, i] * device_sum_bin_x[j, i] / device_profile[j, i]
                device_profile_dip_y[j, i] = device_density_profile[j, i] * device_sum_bin_y[j, i] / device_profile[j, i]
        
        @cuda.jit
        def nan_to_zero_kernel(num_bunch, n_bin, device_profile, device_dip_x, device_dip_y,
                               device_profile_dip_x, device_profile_dip_y):
            """
            Convert NaN values into zeros

            """
            i, j = cuda.grid(2)

            if j < n_bin and i < num_bunch:
                if device_profile[j, i] == 0:
                    device_dip_x[j, i] = 0
                    device_dip_y[j, i] = 0
                    device_profile_dip_x[j, i] = 0
                    device_profile_dip_y[j, i] = 0

        @cuda.jit
        def rw_wake_kernel(num_bunch, n_bin, t0, device_half_d_bin_tau, amp_common, amp_wl_25_integ, amp_wl_long_integ,
                           amp_wt_24_integ, amp_wt_long_integ, device_norm_lim, device_wl_avg, device_wt_avg):
            """
            Calculation of resistive wake functions
            For the short-range wake, we adopt the analytical series expanded equations of Ivanyan and Tsakanov.
            We use average wake functions for each bin by integrating the given wake functions.
            Reference point for determining whether to use short-range or long-range wake function is 11.7*t0.

            """
            i = cuda.grid(1)

            if i < num_bunch:
                if device_half_d_bin_tau[i] >= 11.7*t0:
                    if device_half_d_bin_tau[i] == 11.7*t0:
                        for idx in range(n_bin):
                            if idx == 0:
                                device_wl_avg[n_bin-1+idx, i] = ( wl_25_integ(amp_common, amp_wl_25_integ, 11.7)
                                                / (2*device_half_d_bin_tau[i]) )
                                device_wt_avg[n_bin-1+idx, i] = ( wt_24_integ(amp_common, amp_wt_24_integ, 11.7)
                                                / (2*device_half_d_bin_tau[i]) )
                            else:
                                device_wl_avg[n_bin-1+idx, i] = ( ( wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau[i]))
                                                - wl_long_integ(amp_wl_long_integ, ((2*idx-1)*device_half_d_bin_tau[i])) )
                                                / (2*device_half_d_bin_tau[i]) )
                                device_wt_avg[n_bin-1+idx, i] = ( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau[i]))
                                                - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau[i])) )
                                                / (2*device_half_d_bin_tau[i]) )
                    else:
                        for idx in range(n_bin):
                            if idx == 0:
                                device_wl_avg[n_bin-1+idx, i] = ( wl_25_integ(amp_common, amp_wl_25_integ, 11.7)
                                                + wl_long_integ(amp_wl_long_integ, device_half_d_bin_tau[i])
                                                - wl_long_integ(amp_wl_long_integ, 11.7*t0)
                                                / (2*device_half_d_bin_tau[i]) )
                                device_wt_avg[n_bin-1+idx, i] = ( wt_24_integ(amp_common, amp_wt_24_integ, 11.7)
                                                + wt_long_integ(amp_wt_long_integ, device_half_d_bin_tau[i])
                                                - wt_long_integ(amp_wt_long_integ, 11.7*t0)
                                                / (2*device_half_d_bin_tau[i]) )
                            else:
                                device_wl_avg[n_bin-1+idx, i] = ( ( wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau[i]))
                                                - wl_long_integ(amp_wl_long_integ, ((2*idx-1)*device_half_d_bin_tau[i])) )
                                                / (2*device_half_d_bin_tau[i]) )
                                device_wt_avg[n_bin-1+idx, i] = ( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau[i]))
                                                - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau[i])) )
                                                / (2*device_half_d_bin_tau[i]) )
                else:
                    for idx in range(n_bin):
                        if idx == 0:
                            device_wl_avg[n_bin-1+idx, i] = ( wl_25_integ(amp_common, amp_wl_25_integ, device_norm_lim[i])
                                            / (2*device_half_d_bin_tau[i]) )
                            device_wt_avg[n_bin-1+idx, i] = ( wt_24_integ(amp_common, amp_wt_24_integ, device_norm_lim[i])
                                            / (2*device_half_d_bin_tau[i]) )
                        elif 0 < idx < ( (11.7*t0+device_half_d_bin_tau[i]) // (2*device_half_d_bin_tau[i]) ):
                            device_wl_avg[n_bin-1+idx, i] = ( ( wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx+1)*device_norm_lim[i]))
                                            - wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx-1)*device_norm_lim[i])) )
                                            / (2*device_half_d_bin_tau[i]) )
                            device_wt_avg[n_bin-1+idx, i] = ( ( wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx+1)*device_norm_lim[i]))
                                            - wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx-1)*device_norm_lim[i])) )
                                            / (2*device_half_d_bin_tau[i]) )
                        elif idx == ( (11.7*t0+device_half_d_bin_tau[i]) // (2*device_half_d_bin_tau[i]) ):
                            if ( (11.7*t0+device_half_d_bin_tau[i]) % (2*device_half_d_bin_tau[i]) ) == 0:
                                device_wl_avg[n_bin-1+idx, i] = ( ( wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx+1)*device_norm_lim[i]))
                                            - wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx-1)*device_norm_lim[i])) )
                                            / (2*device_half_d_bin_tau[i]) )
                                device_wt_avg[n_bin-1+idx, i] = ( ( wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx+1)*device_norm_lim[i]))
                                            - wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx-1)*device_norm_lim[i])) )
                                            / (2*device_half_d_bin_tau[i]) )
                            else:
                                device_wl_avg[n_bin-1+idx, i] = ( ( wl_25_integ(amp_common, amp_wl_25_integ, 11.7)
                                                - wl_25_integ(amp_common, amp_wl_25_integ, ((2*idx-1)*device_norm_lim[i]))
                                                + wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau[i]))
                                                - wl_long_integ(amp_wl_long_integ, 11.7*t0) )
                                                / (2*device_half_d_bin_tau[i]) )
                                device_wt_avg[n_bin-1+idx, i] = ( ( wt_24_integ(amp_common, amp_wt_24_integ, 11.7)
                                                - wt_24_integ(amp_common, amp_wt_24_integ, ((2*idx-1)*device_norm_lim[i]))
                                                + wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau[i]))
                                                - wt_long_integ(amp_wt_long_integ, 11.7*t0) )
                                                / (2*device_half_d_bin_tau[i]) )
                        else:
                            device_wl_avg[n_bin-1+idx, i] = ( ( wl_long_integ(amp_wl_long_integ, ((2*idx+1)*device_half_d_bin_tau[i]))
                                            - wl_long_integ(amp_wl_long_integ, ((2*idx-1)*device_half_d_bin_tau[i])) )
                                            / (2*device_half_d_bin_tau[i]) )
                            device_wt_avg[n_bin-1+idx, i] = ( ( wt_long_integ(amp_wt_long_integ, ((2*idx+1)*device_half_d_bin_tau[i]))
                                            - wt_long_integ(amp_wt_long_integ, ((2*idx-1)*device_half_d_bin_tau[i])) )
                                            / (2*device_half_d_bin_tau[i]) )

        @cuda.jit
        def wake_convolution_kernel(num_bunch, n_bin, device_wl_avg, device_wt_avg, device_wp_x, device_wp_y, device_wp_tau,
                                    device_density_profile, device_profile_dip_x, device_profile_dip_y, device_half_d_bin_tau):
            """
            Convolution for wakes

            """
            i, j = cuda.grid(2)

            if (i < num_bunch) and (j >= n_bin - 1) and (j < 2*n_bin - 1):
                for idx in range(n_bin):
                    cuda.atomic.sub(device_wp_tau, (j - n_bin + 1, i),
                                    device_wl_avg[j - idx, i] * device_density_profile[idx, i] * 2 * device_half_d_bin_tau[i])
                    cuda.atomic.add(device_wp_x, (j - n_bin + 1, i),
                                    device_wt_avg[j - idx, i] * device_profile_dip_x[idx, i] * 2 * device_half_d_bin_tau[i])
                    cuda.atomic.add(device_wp_y, (j - n_bin + 1, i),
                                    device_wt_avg[j - idx, i] * device_profile_dip_y[idx, i] * 2 * device_half_d_bin_tau[i])
        
        @cuda.jit
        def wake_interp_kernel(num_bunch, num_particle, n_bin, device_wp_x, device_wp_y, device_wp_tau, device_bin_tau, device_tau,
                               device_wp_x_interp, device_wp_y_interp, device_wp_tau_interp):
            """
            Interpolation of wake potentials

            """
            i, j = cuda.grid(2)

            if i < num_bunch and j < num_particle:
                for idx in range(n_bin-1):
                    if (device_tau[j, i] >= device_bin_tau[idx, i]) and (device_tau[j, i] < device_bin_tau[idx+1, i]):
                        device_wp_x_interp[j, i] = ( (device_wp_x[idx+1, i] - device_wp_x[idx, i]) / (device_bin_tau[idx+1, i] - device_bin_tau[idx, i])
                                                    * (device_tau[j, i] - device_bin_tau[idx, i]) + device_wp_x[idx, i] )
                        device_wp_y_interp[j, i] = ( (device_wp_y[idx+1, i] - device_wp_y[idx, i]) / (device_bin_tau[idx+1, i] - device_bin_tau[idx, i])
                                                    * (device_tau[j, i] - device_bin_tau[idx, i]) + device_wp_y[idx, i] )
                        device_wp_tau_interp[j, i] = ( (device_wp_tau[idx+1, i] - device_wp_tau[idx, i]) / (device_bin_tau[idx+1, i] - device_bin_tau[idx, i])
                                                    * (device_tau[j, i] - device_bin_tau[idx, i]) + device_wp_tau[idx, i] )
                if device_tau[j, i] == device_bin_tau[n_bin-1, i]:
                    device_wp_x_interp[j, i] = device_wp_x[n_bin-1, i]
                    device_wp_y_interp[j, i] = device_wp_y[n_bin-1, i]
                    device_wp_tau_interp[j, i] = device_wp_tau[n_bin-1, i]
        
        @cuda.jit
        def kick_sb_kernel(E0, charge, device_wp_x_interp, device_wp_y_interp, device_wp_tau_interp, device_xp, device_yp, device_delta):
            """
            Kick due to self-bunch wakes

            """
            i, j = cuda.grid(2)
            
            cuda.atomic.add(device_xp, (j, i), device_wp_x_interp[j, i] * charge / E0)
            cuda.atomic.add(device_yp, (j, i), device_wp_y_interp[j, i] * charge / E0)
            cuda.atomic.add(device_delta, (j, i), device_wp_tau_interp[j, i] * charge / E0)
        
        @cuda.jit
        def shift_tables_kernel(num_bunch, turns_lrrw, T0, device_tau_lrrw, device_x_lrrw, device_y_lrrw):
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
                # This operations corresponds to numpy.roll().
                idx = (j - 1) % turns_lrrw
                device_tau_lrrw[j, i] = device_tau_lrrw[idx, i] + T0
                device_x_lrrw[j, i] = device_x_lrrw[idx, i]
                device_y_lrrw[j, i] = device_y_lrrw[idx, i]

        @cuda.jit
        def mean_ps1_kernel(device_x, device_y, device_tau, device_1st_sum_tau_lrrw, device_1st_sum_x_lrrw, device_1st_sum_y_lrrw,
                            num_bunch):
            """
            1st prefix sum for the calculation of mean values (tau, x, y) for each bunch
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y

            sum_tau_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_x_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_y_shared = cuda.shared.array(threadperblock, numba.float32)
            
            sum_tau_shared[local_j, local_i] = device_tau[j, i]
            sum_x_shared[local_j, local_i] = device_x[j, i]
            sum_y_shared[local_j, local_i] = device_y[j, i]
            cuda.syncthreads()

            s = threadperblock[1]
            s >>= 1
            while s > 0:
                if local_j < s and local_i < num_bunch:
                    sum_tau_shared[local_j, local_i] += sum_tau_shared[local_j + s, local_i]
                    sum_x_shared[local_j, local_i] += sum_x_shared[local_j + s, local_i]
                    sum_y_shared[local_j, local_i] += sum_y_shared[local_j + s, local_i]
                cuda.syncthreads()

            if local_j == 0 and i < num_bunch:
                device_1st_sum_tau_lrrw[cuda.blockIdx.y, i] = sum_tau_shared[0, local_i]
                device_1st_sum_x_lrrw[cuda.blockIdx.y, i] = sum_x_shared[0, local_i]
                device_1st_sum_y_lrrw[cuda.blockIdx.y, i] = sum_y_shared[0, local_i]
        
        @cuda.jit
        def mean_ps2_kernel(device_1st_sum_tau_lrrw, device_1st_sum_x_lrrw, device_1st_sum_y_lrrw, device_2nd_sum_tau_lrrw,
                            device_2nd_sum_x_lrrw, device_2nd_sum_y_lrrw, num_bunch):
            """
            2nd prefix sum for the calculation of mean values (tau, x, y) for each bunch
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y

            sum_tau_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_x_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_y_shared = cuda.shared.array(threadperblock, numba.float32)
            
            sum_tau_shared[local_j, local_i] = device_1st_sum_tau_lrrw[j, i]
            sum_x_shared[local_j, local_i] = device_1st_sum_x_lrrw[j, i]
            sum_y_shared[local_j, local_i] = device_1st_sum_y_lrrw[j, i]
            cuda.syncthreads()

            s = threadperblock[1]
            s >>= 1
            while s > 0:
                if local_j < s and local_i < num_bunch:
                    sum_tau_shared[local_j, local_i] += sum_tau_shared[local_j + s, local_i]
                    sum_x_shared[local_j, local_i] += sum_x_shared[local_j + s, local_i]
                    sum_y_shared[local_j, local_i] += sum_y_shared[local_j + s, local_i]
                cuda.syncthreads()

            if local_j == 0 and i < num_bunch:
                device_2nd_sum_tau_lrrw[cuda.blockIdx.y, i] = sum_tau_shared[0, local_i]
                device_2nd_sum_x_lrrw[cuda.blockIdx.y, i] = sum_x_shared[0, local_i]
                device_2nd_sum_y_lrrw[cuda.blockIdx.y, i] = sum_y_shared[0, local_i]

        @cuda.jit
        def mean_as_kernel(device_2nd_sum_tau_lrrw, device_2nd_sum_x_lrrw, device_2nd_sum_y_lrrw, device_axis_sum_tau_lrrw,
                           device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, num_bunch, num_red):
            """
            Axis sum for the calculation of mean values (tau, x, y) for each bunch
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i = cuda.grid(1)

            if i < num_bunch:
                for idx in range(num_red):
                    cuda.atomic.add(device_axis_sum_tau_lrrw, i, device_2nd_sum_tau_lrrw[idx, i])
                    cuda.atomic.add(device_axis_sum_x_lrrw, i, device_2nd_sum_x_lrrw[idx, i])
                    cuda.atomic.add(device_axis_sum_y_lrrw, i, device_2nd_sum_y_lrrw[idx, i])
        
        @cuda.jit
        def mean_tables_kernel(T1, num_bunch, num_particle, device_axis_sum_tau_lrrw, device_axis_sum_x_lrrw,
                               device_axis_sum_y_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw):
            """
            Axis sum for the calculation of mean values (tau, x, y) for each bunch
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i = cuda.grid(1)
            
            if i < num_bunch:
                device_tau_lrrw[0, i] = device_axis_sum_tau_lrrw[i] / num_particle - i*T1
                device_x_lrrw[0, i] = device_axis_sum_x_lrrw[i] / num_particle
                device_y_lrrw[0, i] = device_axis_sum_y_lrrw[i] / num_particle
        
        @cuda.jit
        def get_kick_btb_kernel(num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw,
                                device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau,
                                charge, amp_wl_long, amp_wt_long):
            """
            Preparation of bunch to bunch kick
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)

            if j < turns_lrrw and i < num_bunch:
                # idx is the test bunch index.
                for idx in range(num_bunch):
                    if not isnan(device_tau_lrrw[0, idx]):
                        if j == 0 and idx <= i:
                            continue
                        if not isnan(device_tau_lrrw[j, i]):
                            cuda.atomic.sub(device_sum_kick_tau, idx, wl_long( amp_wl_long, (device_tau_lrrw[j, i] - device_tau_lrrw[0, idx]) )*charge)
                            cuda.atomic.add(device_sum_kick_x, idx, wt_long( amp_wt_long, (device_tau_lrrw[j, i] - device_tau_lrrw[0, idx]) )*device_x_lrrw[j, i]*charge)
                            cuda.atomic.add(device_sum_kick_y, idx, wt_long( amp_wt_long, (device_tau_lrrw[j, i] - device_tau_lrrw[0, idx]) )*device_y_lrrw[j, i]*charge)
                        else:
                            pass
                    else:
                        pass
        
        @cuda.jit
        def kick_btb_kernel(num_bunch, num_particle, device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau, E0):
            """
            Calculation of bunch to bunch kick
            This is one of several kernels used to calculate the long-range resistive wall wake.
            """
            i, j = cuda.grid(2)

            if j < num_particle and i < num_bunch:
                for idx in range(num_bunch):
                    cuda.atomic.add(device_xp, (j, i), device_sum_kick_tau[idx] / E0)
                    cuda.atomic.add(device_yp, (j, i), device_sum_kick_x[idx] / E0)
                    cuda.atomic.add(device_delta, (j, i), device_sum_kick_y[idx] / E0)
        
        @cuda.jit
        def monitor_ps1_kernel(device_x, device_xp, device_y, device_yp, device_tau, device_delta,
                               device_1st_sum_x_squared, device_1st_sum_xp_squared, device_1st_sum_x_xp,
                               device_1st_sum_y_squared, device_1st_sum_yp_squared, device_1st_sum_y_yp,
                               device_1st_sum_tau_squared, device_1st_sum_delta_squared, device_1st_sum_tau_delta,
                               device_1st_sum_tau, num_bunch):
            """
            1st prefix sum for monitor
            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y

            sum_x_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_xp_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_x_xp_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_y_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_yp_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_y_yp_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_tau_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_delta_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_tau_delta_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_tau_shared = cuda.shared.array(threadperblock, numba.float32)

            sum_x_squared_shared[local_j, local_i] = device_x[j, i]**2
            sum_xp_squared_shared[local_j, local_i] = device_xp[j, i]**2
            sum_x_xp_shared[local_j, local_i] = device_x[j, i] * device_xp[j, i] 
            sum_y_squared_shared[local_j, local_i] = device_y[j, i]**2
            sum_yp_squared_shared[local_j, local_i] = device_yp[j, i]**2
            sum_y_yp_shared[local_j, local_i] = device_y[j, i] * device_yp[j, i]
            sum_tau_squared_shared[local_j, local_i] = device_tau[j, i]**2
            sum_delta_squared_shared[local_j, local_i] = device_delta[j, i]**2
            sum_tau_delta_shared[local_j, local_i] = device_tau[j, i] * device_delta[j, i]
            sum_tau_shared[local_j, local_i] = device_tau[j, i]
            cuda.syncthreads()

            s = threadperblock[1]
            s >>= 1
            while s > 0:
                if local_j < s and local_i < num_bunch:
                    sum_x_squared_shared[local_j, local_i] += sum_x_squared_shared[local_j + s, local_i]
                    sum_xp_squared_shared[local_j, local_i] += sum_xp_squared_shared[local_j + s, local_i]
                    sum_x_xp_shared[local_j, local_i] += sum_x_xp_shared[local_j + s, local_i]
                    sum_y_squared_shared[local_j, local_i] += sum_y_squared_shared[local_j + s, local_i]
                    sum_yp_squared_shared[local_j, local_i] += sum_yp_squared_shared[local_j + s, local_i]
                    sum_y_yp_shared[local_j, local_i] += sum_y_yp_shared[local_j + s, local_i]
                    sum_tau_squared_shared[local_j, local_i] += sum_tau_squared_shared[local_j + s, local_i]
                    sum_delta_squared_shared[local_j, local_i] += sum_delta_squared_shared[local_j + s, local_i]
                    sum_tau_delta_shared[local_j, local_i] += sum_tau_delta_shared[local_j + s, local_i]
                    sum_tau_shared[local_j, local_i] += sum_tau_shared[local_j + s, local_i]
                cuda.syncthreads()
                s >>= 1
                
            if local_j == 0 and i < num_bunch:
                device_1st_sum_x_squared[cuda.blockIdx.y, i] = sum_x_squared_shared[0, local_i]
                device_1st_sum_xp_squared[cuda.blockIdx.y, i] = sum_xp_squared_shared[0, local_i]
                device_1st_sum_x_xp[cuda.blockIdx.y, i] = sum_x_xp_shared[0, local_i]
                device_1st_sum_y_squared[cuda.blockIdx.y, i] = sum_y_squared_shared[0, local_i]
                device_1st_sum_yp_squared[cuda.blockIdx.y, i] = sum_yp_squared_shared[0, local_i]
                device_1st_sum_y_yp[cuda.blockIdx.y, i] = sum_y_yp_shared[0, local_i]
                device_1st_sum_tau_squared[cuda.blockIdx.y, i] = sum_tau_squared_shared[0, local_i]
                device_1st_sum_delta_squared[cuda.blockIdx.y, i] = sum_delta_squared_shared[0, local_i]
                device_1st_sum_tau_delta[cuda.blockIdx.y, i] = sum_tau_delta_shared[0, local_i]
                device_1st_sum_tau[cuda.blockIdx.y, i] = sum_tau_shared[0, local_i]

        @cuda.jit
        def monitor_ps2_kernel(device_1st_sum_x_squared, device_1st_sum_xp_squared, device_1st_sum_x_xp, device_1st_sum_y_squared,
                              device_1st_sum_yp_squared, device_1st_sum_y_yp, device_1st_sum_tau_squared, device_1st_sum_delta_squared,
                              device_1st_sum_tau_delta, device_1st_sum_tau, device_2nd_sum_x_squared, device_2nd_sum_xp_squared,
                              device_2nd_sum_x_xp, device_2nd_sum_y_squared, device_2nd_sum_yp_squared, device_2nd_sum_y_yp,
                              device_2nd_sum_tau_squared, device_2nd_sum_delta_squared, device_2nd_sum_tau_delta, device_2nd_sum_tau, num_bunch):
            """
            2nd prefix sum for monitor

            """
            i, j = cuda.grid(2)
            local_i, local_j = cuda.threadIdx.x, cuda.threadIdx.y

            sum_x_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_xp_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_x_xp_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_y_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_yp_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_y_yp_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_tau_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_delta_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_tau_delta_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_tau_shared = cuda.shared.array(threadperblock, numba.float32)

            sum_x_squared_shared[local_j, local_i] = device_1st_sum_x_squared[j, i]
            sum_xp_squared_shared[local_j, local_i] = device_1st_sum_xp_squared[j, i]
            sum_x_xp_shared[local_j, local_i] = device_1st_sum_x_xp[j, i]
            sum_y_squared_shared[local_j, local_i] = device_1st_sum_y_squared[j, i]
            sum_yp_squared_shared[local_j, local_i] = device_1st_sum_yp_squared[j, i]
            sum_y_yp_shared[local_j, local_i] = device_1st_sum_y_yp[j, i]
            sum_tau_squared_shared[local_j, local_i] = device_1st_sum_tau_squared[j, i]
            sum_delta_squared_shared[local_j, local_i] = device_1st_sum_delta_squared[j, i]
            sum_tau_delta_shared[local_j, local_i] = device_1st_sum_tau_delta[j, i]
            sum_tau_shared[local_j, local_i] = device_1st_sum_tau[j, i]
            cuda.syncthreads()

            s = threadperblock[1]
            s >>= 1
            while s > 0:
                if local_j < s and local_i < num_bunch:
                    sum_x_squared_shared[local_j, local_i] += sum_x_squared_shared[local_j + s, local_i]
                    sum_xp_squared_shared[local_j, local_i] += sum_xp_squared_shared[local_j + s, local_i]
                    sum_x_xp_shared[local_j, local_i] += sum_x_xp_shared[local_j + s, local_i]
                    sum_y_squared_shared[local_j, local_i] += sum_y_squared_shared[local_j + s, local_i]
                    sum_yp_squared_shared[local_j, local_i] += sum_yp_squared_shared[local_j + s, local_i]
                    sum_y_yp_shared[local_j, local_i] += sum_y_yp_shared[local_j + s, local_i]
                    sum_tau_squared_shared[local_j, local_i] += sum_tau_squared_shared[local_j + s, local_i]
                    sum_delta_squared_shared[local_j, local_i] += sum_delta_squared_shared[local_j + s, local_i]
                    sum_tau_delta_shared[local_j, local_i] += sum_tau_delta_shared[local_j + s, local_i]
                    sum_tau_shared[local_j, local_i] += sum_tau_shared[local_j + s, local_i]
                cuda.syncthreads()
                s >>= 1
                
            if local_j == 0 and i < num_bunch:
                    device_2nd_sum_x_squared[cuda.blockIdx.y, i] = sum_x_squared_shared[0, local_i]
                    device_2nd_sum_xp_squared[cuda.blockIdx.y, i] = sum_xp_squared_shared[0, local_i]
                    device_2nd_sum_x_xp[cuda.blockIdx.y, i] = sum_x_xp_shared[0, local_i]
                    device_2nd_sum_y_squared[cuda.blockIdx.y, i] = sum_y_squared_shared[0, local_i]
                    device_2nd_sum_yp_squared[cuda.blockIdx.y, i] = sum_yp_squared_shared[0, local_i]
                    device_2nd_sum_y_yp[cuda.blockIdx.y, i] = sum_y_yp_shared[0, local_i]
                    device_2nd_sum_tau_squared[cuda.blockIdx.y, i] = sum_tau_squared_shared[0, local_i]
                    device_2nd_sum_delta_squared[cuda.blockIdx.y, i] = sum_delta_squared_shared[0, local_i]
                    device_2nd_sum_tau_delta[cuda.blockIdx.y, i] = sum_tau_delta_shared[0, local_i]
                    device_2nd_sum_tau[cuda.blockIdx.y, i] = sum_tau_shared[0, local_i]

        @cuda.jit
        def monitor_as_kernel(device_2nd_sum_x_squared, device_2nd_sum_xp_squared, device_2nd_sum_x_xp, device_2nd_sum_y_squared,
                              device_2nd_sum_yp_squared, device_2nd_sum_y_yp, device_2nd_sum_tau_squared, device_2nd_sum_delta_squared,
                              device_2nd_sum_tau_delta, device_2nd_sum_tau, device_axis_sum_x_squared, device_axis_sum_xp_squared,
                              device_axis_sum_x_xp, device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                              device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau_delta, device_axis_sum_tau,
                              num_bunch, num_red):
            """
            Axis sum for monitor

            """
            i = cuda.grid(1)

            if i < num_bunch:
                for idx in range(num_red):
                    cuda.atomic.add(device_axis_sum_x_squared, i, device_2nd_sum_x_squared[idx, i])
                    cuda.atomic.add(device_axis_sum_xp_squared, i, device_2nd_sum_xp_squared[idx, i])
                    cuda.atomic.add(device_axis_sum_x_xp, i, device_2nd_sum_x_xp[idx, i])
                    cuda.atomic.add(device_axis_sum_y_squared, i, device_2nd_sum_y_squared[idx, i])
                    cuda.atomic.add(device_axis_sum_yp_squared, i, device_2nd_sum_yp_squared[idx, i])
                    cuda.atomic.add(device_axis_sum_y_yp, i, device_2nd_sum_y_yp[idx, i])
                    cuda.atomic.add(device_axis_sum_tau_squared, i, device_2nd_sum_tau_squared[idx, i])
                    cuda.atomic.add(device_axis_sum_delta_squared, i, device_2nd_sum_delta_squared[idx, i])
                    cuda.atomic.add(device_axis_sum_tau_delta, i, device_2nd_sum_tau_delta[idx, i])
                    cuda.atomic.add(device_axis_sum_tau, i, device_2nd_sum_tau[idx, i])

        @cuda.jit
        def monitor_results_kernel(device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                              device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                              device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau_delta,
                              device_axis_sum_tau, device_beam_emitX, device_beam_emitY, device_beam_emitS,
                              device_bunch_length, num_bunch, num_particle, k):
            """
            Final results

            """
            i = cuda.grid(1)
            
            if i < num_bunch:
                device_beam_emitX[k, i] = sqrt( (device_axis_sum_x_squared[i] * device_axis_sum_xp_squared[i] - device_axis_sum_x_xp[i]**2) / (num_particle**2) )
                device_beam_emitY[k, i] = sqrt( (device_axis_sum_y_squared[i] * device_axis_sum_yp_squared[i] - device_axis_sum_y_yp[i]**2) / (num_particle**2) )
                device_beam_emitS[k, i] = sqrt( (device_axis_sum_tau_squared[i] * device_axis_sum_delta_squared[i] - device_axis_sum_tau_delta[i]**2) / (num_particle**2) )
                device_bunch_length[k, i] = sqrt( (device_axis_sum_tau_squared[i]/num_particle) - (device_axis_sum_tau[i]/num_particle)**2 )

        if isinstance(bunch, Beam):
            beam = bunch
            # num_bunch = beam.__len__()
            num_bunch = self.ring.h
            num_particle = beam[0].mp_number
            charge = beam[0].charge
            charge_per_mp = beam[0].charge_per_mp

            x = np.empty((num_particle, num_bunch), dtype="f4")
            xp = np.empty((num_particle, num_bunch), dtype="f4")
            y = np.empty((num_particle, num_bunch), dtype="f4")
            yp = np.empty((num_particle, num_bunch), dtype="f4")
            tau = np.empty((num_particle, num_bunch), dtype="f4")
            delta = np.empty((num_particle, num_bunch), dtype="f4")

            # density_profile = np.empty((self.n_bin, num_bunch), dtype="f4")
            profile = np.empty((self.n_bin, num_bunch), dtype="f4")
            dip_x = np.empty((self.n_bin, num_bunch), dtype="f4")
            dip_y = np.empty((self.n_bin, num_bunch), dtype="f4")

            wp_tau = np.empty((self.n_bin, num_bunch), dtype="f4")
            wp_x = np.empty((self.n_bin, num_bunch), dtype="f4")
            wp_y = np.empty((self.n_bin, num_bunch), dtype="f4")
            
            wp_x_interp = np.empty((num_particle, num_bunch), dtype="f4")
            wp_y_interp = np.empty((num_particle, num_bunch), dtype="f4")
            wp_tau_interp = np.empty((num_particle, num_bunch), dtype="f4")

            bin_tau = np.empty((self.n_bin, num_bunch), dtype="f4")

            axis_min_tau = np.empty((num_bunch), dtype="f4")
            half_d_bin_tau = np.empty((num_bunch), dtype="f4")
            t = np.empty((self.n_bin, num_bunch), dtype="f4")

            beam_emitX = np.empty((turns, num_bunch), dtype="f4")
            beam_emitY = np.empty((turns, num_bunch), dtype="f4")
            beam_emitS = np.empty((turns, num_bunch), dtype="f4")
            bunch_length = np.empty((turns, num_bunch), dtype="f4")

            tau_lrrw = np.ones((turns_lrrw, num_bunch), dtype="f4") * np.inf
            x_lrrw = np.zeros((turns_lrrw, num_bunch), dtype="f4")
            y_lrrw = np.zeros((turns_lrrw, num_bunch), dtype="f4")
            charge_lrrw = np.zeros((turns_lrrw, num_bunch), dtype="f4")

            sigma_xp = self.ring.sigma()[1]
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
            t0 = (2*self.rho*self.radius**2/Z0)**(1/3) / c
            print(t0)

            if 11.7*t0 > self.ring.T1:
                raise ValueError("The approximated wake functions are not valid.")

            amp_common = 0.5*sqrt(2/pi)
            amp_wl_25_integ = (Z0*c*t0) / (pi*self.radius**2) * self.length
            amp_wl_long_integ = sqrt(Z0 * self.rho / (c * pi)) / (2*pi*self.radius) * self.length
            amp_wt_24_integ = (Z0*c**2*t0**2) / (pi*self.radius**4) * self.length
            amp_wt_long_integ = 2 * sqrt(Z0*c*self.rho / pi) / (pi*self.radius**3) * self.length
            amp_wl_long = -1 * sqrt(Z0*self.rho / (c*pi)) / (4*pi*self.radius) * self.length
            amp_wt_long = sqrt(Z0*c*self.rho / pi) / (pi*self.radius**3) * self.length

            for bunch_index, bunch_ref in enumerate(beam):
                x[:, bunch_index] = bunch_ref["x"]
                xp[:, bunch_index] = bunch_ref["xp"]
                y[:, bunch_index] = bunch_ref["y"]
                yp[:, bunch_index] = bunch_ref["yp"]
                tau[:, bunch_index] = bunch_ref["tau"]
                delta[:, bunch_index] = bunch_ref["delta"]

            threadperblock_x = 16
            threadperblock_y = 16
            threadperblock = (threadperblock_x, threadperblock_y)
            blockpergrid = (num_bunch // threadperblock_x + 1, num_particle // threadperblock_y + 1)
            blockpergrid_red = (blockpergrid[0], blockpergrid[1] // threadperblock_y + 1)
            num_red = blockpergrid_red[1]
            blockpergrid_pad = (blockpergrid[0], (2*self.n_bin-1) // threadperblock_y + 1)
            blockpergrid_bin = (blockpergrid[0], self.n_bin // threadperblock_y + 1)
            blockpergrid_lrrw = (blockpergrid[0], turns_lrrw // threadperblock_y + 1)

            rng_states = create_xoroshiro128p_states(num_particle*num_bunch, seed=1) # time.time()

            # Calculations in GPU
            # Pin memory
            with cuda.pinned(x, xp, y, yp, tau, delta, x_lrrw, y_lrrw, tau_lrrw):
                
                # Create a CUDA stream
                stream = cuda.stream()
                
                device_x = cuda.to_device(x, stream=stream)
                device_xp = cuda.to_device(xp, stream=stream)
                device_y = cuda.to_device(y, stream=stream)
                device_yp = cuda.to_device(yp, stream=stream)
                device_tau = cuda.to_device(tau, stream=stream)
                device_delta = cuda.to_device(delta, stream=stream)

                device_x_lrrw = cuda.to_device(x_lrrw, stream=stream)
                device_y_lrrw = cuda.to_device(y_lrrw, stream=stream)
                device_tau_lrrw = cuda.to_device(tau_lrrw, stream=stream)

                device_rand_xp = cuda.device_array((num_particle, num_bunch), dtype=np.float32, stream=stream)
                device_rand_yp = cuda.device_array((num_particle, num_bunch), dtype=np.float32, stream=stream)
                device_rand_delta = cuda.device_array((num_particle, num_bunch), dtype=np.float32, stream=stream)

                device_1st_min_tau = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_max_tau = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                
                device_2nd_min_tau = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_max_tau = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                
                device_axis_min_tau = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_max_tau = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)

                device_bin_tau = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_half_d_bin_tau = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_t = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                # device_norm_t = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_norm_lim = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)

                device_wl_avg = cuda.device_array((2*self.n_bin-1, num_bunch), dtype=np.float32, stream=stream)
                device_wt_avg = cuda.device_array((2*self.n_bin-1, num_bunch), dtype=np.float32, stream=stream)

                device_wp_x = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_wp_y = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_wp_tau = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)

                device_wp_x_interp = cuda.device_array((num_particle, num_bunch), dtype=np.float32, stream=stream)
                device_wp_y_interp = cuda.device_array((num_particle, num_bunch), dtype=np.float32, stream=stream)
                device_wp_tau_interp = cuda.device_array((num_particle, num_bunch), dtype=np.float32, stream=stream)

                # device_sorted_idx = cuda.device_array((num_particle, num_bunch), dtype=np.int32, stream=stream)
                device_density_profile = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_profile = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_sum_bin_x = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_sum_bin_y = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_dip_x = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_dip_y = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_profile_dip_x = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)
                device_profile_dip_y = cuda.device_array((self.n_bin, num_bunch), dtype=np.float32, stream=stream)

                device_1st_sum_x_squared = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_xp_squared = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_x_xp = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_y_squared = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_yp_squared = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_y_yp = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_tau_squared = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_delta_squared = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_tau_delta = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_tau = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                
                device_1st_sum_x_lrrw = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_y_lrrw = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)
                device_1st_sum_tau_lrrw = cuda.device_array((blockpergrid[1], num_bunch), dtype=np.float32, stream=stream)

                device_2nd_sum_x_squared = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_xp_squared = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_x_xp = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_y_squared = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_yp_squared = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_y_yp = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_tau_squared = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_delta_squared = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_tau_delta = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_tau = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)

                device_2nd_sum_x_lrrw = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_y_lrrw = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)
                device_2nd_sum_tau_lrrw = cuda.device_array((blockpergrid_red[1], num_bunch), dtype=np.float32, stream=stream)

                device_axis_sum_x_squared = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_xp_squared = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_x_xp = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_y_squared = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_yp_squared = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_y_yp = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_tau_squared = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_delta_squared = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_tau_delta = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_tau = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)

                device_axis_sum_x_lrrw = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_y_lrrw = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_axis_sum_tau_lrrw = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)

                device_sum_kick_x = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_sum_kick_y = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)
                device_sum_kick_tau = cuda.device_array((num_bunch,), dtype=np.float32, stream=stream)

                device_beam_emitX = cuda.device_array_like(beam_emitX, stream=stream)
                device_beam_emitY = cuda.device_array_like(beam_emitY, stream=stream)
                device_beam_emitS = cuda.device_array_like(beam_emitS, stream=stream)
                device_bunch_length = cuda.device_array_like(bunch_length, stream=stream)
                
                for k in range(turns):
                    rng_kernel[blockpergrid, threadperblock, stream](num_particle, num_bunch, rng_states, device_rand_xp, device_rand_yp, device_rand_delta)

                    map_kernel[blockpergrid, threadperblock, stream](device_x, device_xp, device_y, device_yp, device_tau, device_delta,
                                                                     self.ring.U0, self.ring.E0, self.ring.ac, self.ring.T0, self.ring.sigma_delta,
                                                                     self.ring.omega1, sigma_xp, sigma_yp, tau_h, tau_v, tau_l, dispersion_x, dispersion_xp,
                                                                     dispersion_y, dispersion_yp, self.m, self.Vc, self.theta, tune_x, tune_y, chro_x, chro_y, pi,
                                                                     alpha_x, alpha_y, beta_x, beta_y, gamma_x, gamma_y, device_rand_xp, device_rand_yp, device_rand_delta,
                                                                     culm, cutm, cusr, curfc, curw, cubm)
                    
                    mm_pr1_kernel[blockpergrid, threadperblock, stream](device_tau, device_1st_min_tau, device_1st_max_tau, num_bunch)

                    mm_pr2_kernel[blockpergrid_red, threadperblock, stream](device_1st_min_tau, device_1st_max_tau, device_2nd_min_tau, device_2nd_max_tau, num_bunch)

                    initialize_gm_kernel[blockpergrid_pad, threadperblock, stream](device_2nd_min_tau, device_2nd_max_tau, device_axis_min_tau, device_axis_max_tau,
                                                                                   device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp,
                                                                                   device_axis_sum_y_squared, device_axis_sum_yp_squared, device_axis_sum_y_yp,
                                                                                   device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau_delta,
                                                                                   device_axis_sum_tau, device_density_profile, device_profile, device_sum_bin_x, device_sum_bin_y,
                                                                                   device_wl_avg, device_wt_avg, device_wp_x, device_wp_y, device_wp_tau,
                                                                                   device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, device_axis_sum_tau_lrrw, device_sum_kick_x,
                                                                                   device_sum_kick_y, device_sum_kick_tau, num_bunch, self.n_bin, k)

                    mm_results_kernel[blockpergrid[0], threadperblock[0], stream](device_2nd_min_tau, device_2nd_max_tau, device_axis_min_tau, device_axis_max_tau, num_bunch, num_red)

                    binning_kernel[blockpergrid[0], threadperblock[0], stream](num_bunch, self.n_bin, device_axis_min_tau, device_axis_max_tau, device_bin_tau, device_half_d_bin_tau,
                                                                               t0, device_norm_lim, device_t)

                    sorting_kernel[blockpergrid, threadperblock, stream](num_bunch, num_particle, self.n_bin, device_tau, device_half_d_bin_tau, device_bin_tau,
                                                                         device_density_profile, device_profile, device_x, device_y, device_sum_bin_x, device_sum_bin_y,
                                                                         charge, charge_per_mp)
                    
                    dipole_moment_kernel[blockpergrid_bin, threadperblock, stream](num_bunch, self.n_bin, device_profile, device_density_profile, device_sum_bin_x, device_sum_bin_y,
                                                                                   device_dip_x, device_dip_y, device_profile_dip_x, device_profile_dip_y)

                    nan_to_zero_kernel[blockpergrid_bin, threadperblock, stream](num_bunch, self.n_bin, device_profile, device_dip_x, device_dip_y, device_profile_dip_x,
                                                                                 device_profile_dip_y)

                    rw_wake_kernel[blockpergrid[0], threadperblock[0], stream](num_bunch, self.n_bin, t0, device_half_d_bin_tau, amp_common, amp_wl_25_integ, amp_wl_long_integ,
                                                                               amp_wt_24_integ, amp_wt_long_integ, device_norm_lim, device_wl_avg, device_wt_avg)

                    wake_convolution_kernel[blockpergrid_pad, threadperblock, stream](num_bunch, self.n_bin, device_wl_avg, device_wt_avg, device_wp_x, device_wp_y, device_wp_tau,
                                                                                     device_density_profile, device_profile_dip_x, device_profile_dip_y, device_half_d_bin_tau)
                    
                    wake_interp_kernel[blockpergrid, threadperblock, stream](num_bunch, num_particle, self.n_bin, device_wp_x, device_wp_y, device_wp_tau, device_bin_tau, device_tau,
                                                                            device_wp_x_interp, device_wp_y_interp, device_wp_tau_interp)
                    
                    kick_sb_kernel[blockpergrid, threadperblock, stream](self.ring.E0, charge, device_wp_x_interp, device_wp_y_interp, device_wp_tau_interp, device_xp, device_yp, device_delta)
                    
                    if k > 0:
                        shift_tables_kernel[blockpergrid_lrrw, threadperblock, stream](num_bunch, turns_lrrw, self.ring.T0, device_tau_lrrw, device_x_lrrw, device_y_lrrw)

                    mean_ps1_kernel[blockpergrid, threadperblock, stream](device_x, device_y, device_tau, device_1st_sum_tau_lrrw, device_1st_sum_x_lrrw, device_1st_sum_y_lrrw, num_bunch)

                    mean_ps2_kernel[blockpergrid_red, threadperblock, stream](device_1st_sum_tau_lrrw, device_1st_sum_x_lrrw, device_1st_sum_y_lrrw, device_2nd_sum_tau_lrrw,
                                                                              device_2nd_sum_x_lrrw, device_2nd_sum_y_lrrw, num_bunch)

                    mean_as_kernel[blockpergrid[0], threadperblock[0], stream](device_2nd_sum_tau_lrrw, device_2nd_sum_x_lrrw, device_2nd_sum_y_lrrw, device_axis_sum_tau_lrrw,
                                                                               device_axis_sum_x_lrrw, device_axis_sum_y_lrrw, num_bunch, num_red)

                    mean_tables_kernel[blockpergrid[0], threadperblock[0], stream](self.ring.T1, num_bunch, num_particle, device_axis_sum_tau_lrrw, device_axis_sum_x_lrrw, device_axis_sum_y_lrrw,
                                                                                  device_tau_lrrw, device_x_lrrw, device_y_lrrw)
                    
                    get_kick_btb_kernel[blockpergrid_lrrw, threadperblock, stream](num_bunch, turns_lrrw, device_tau_lrrw, device_x_lrrw, device_y_lrrw, device_sum_kick_x, device_sum_kick_y,
                                                                                   device_sum_kick_tau, charge, amp_wl_long, amp_wt_long)
                    
                    kick_btb_kernel[blockpergrid, threadperblock, stream](num_bunch, num_particle, device_sum_kick_x, device_sum_kick_y, device_sum_kick_tau, device_tau,
                                                                          device_x, device_y, self.ring.E0)

                    monitor_ps1_kernel[blockpergrid, threadperblock, stream](device_x, device_xp, device_y, device_yp, device_tau, device_delta,
                                                                             device_1st_sum_x_squared, device_1st_sum_xp_squared, device_1st_sum_x_xp, device_1st_sum_y_squared,
                                                                             device_1st_sum_yp_squared, device_1st_sum_y_yp, device_1st_sum_tau_squared, device_1st_sum_delta_squared,
                                                                             device_1st_sum_tau_delta, device_1st_sum_tau, num_bunch)
                    
                    monitor_ps2_kernel[blockpergrid_red, threadperblock, stream](device_1st_sum_x_squared, device_1st_sum_xp_squared, device_1st_sum_x_xp, device_1st_sum_y_squared,
                                                       device_1st_sum_yp_squared, device_1st_sum_y_yp, device_1st_sum_tau_squared, device_1st_sum_delta_squared, device_1st_sum_tau_delta,
                                                       device_1st_sum_tau, device_2nd_sum_x_squared, device_2nd_sum_xp_squared, device_2nd_sum_x_xp, device_2nd_sum_y_squared,
                                                       device_2nd_sum_yp_squared, device_2nd_sum_y_yp, device_2nd_sum_tau_squared, device_2nd_sum_delta_squared, device_2nd_sum_tau_delta,
                                                       device_2nd_sum_tau, num_bunch)

                    monitor_as_kernel[blockpergrid[0], threadperblock[0], stream](device_2nd_sum_x_squared, device_2nd_sum_xp_squared, device_2nd_sum_x_xp, device_2nd_sum_y_squared,
                                                device_2nd_sum_yp_squared, device_2nd_sum_y_yp, device_2nd_sum_tau_squared, device_2nd_sum_delta_squared, device_2nd_sum_tau_delta,
                                                device_2nd_sum_tau, device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared,
                                                device_axis_sum_yp_squared, device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau_delta,
                                                device_axis_sum_tau, num_bunch, num_red)

                    monitor_results_kernel[blockpergrid[0], threadperblock[0], stream](device_axis_sum_x_squared, device_axis_sum_xp_squared, device_axis_sum_x_xp, device_axis_sum_y_squared,
                                                device_axis_sum_yp_squared, device_axis_sum_y_yp, device_axis_sum_tau_squared, device_axis_sum_delta_squared, device_axis_sum_tau_delta,
                                                device_axis_sum_tau, device_beam_emitX, device_beam_emitY, device_beam_emitS, device_bunch_length, num_bunch, num_particle, k)

                device_x.copy_to_host(x, stream=stream)
                device_xp.copy_to_host(xp, stream=stream)
                device_y.copy_to_host(y, stream=stream)
                device_yp.copy_to_host(yp, stream=stream)
                device_tau.copy_to_host(tau, stream=stream)
                device_delta.copy_to_host(delta, stream=stream)

                device_beam_emitX.copy_to_host(beam_emitX, stream=stream)
                device_beam_emitY.copy_to_host(beam_emitY, stream=stream)
                device_beam_emitS.copy_to_host(beam_emitS, stream=stream)
                device_bunch_length.copy_to_host(bunch_length, stream=stream)

                device_wp_tau.copy_to_host(wp_tau, stream=stream)
                device_wp_x.copy_to_host(wp_x, stream=stream)
                device_wp_y.copy_to_host(wp_y, stream=stream)
                
                device_wp_tau_interp.copy_to_host(wp_tau_interp, stream=stream)
                device_wp_x_interp.copy_to_host(wp_x_interp, stream=stream)
                device_wp_y_interp.copy_to_host(wp_y_interp, stream=stream)

                device_profile.copy_to_host(profile, stream=stream)
                device_dip_x.copy_to_host(dip_x, stream=stream)
                device_dip_y.copy_to_host(dip_y, stream=stream)
                
                device_axis_min_tau.copy_to_host(axis_min_tau, stream=stream)
                device_half_d_bin_tau.copy_to_host(half_d_bin_tau, stream=stream)
                device_t.copy_to_host(t, stream=stream)

                device_bin_tau.copy_to_host(bin_tau, stream=stream)

                # device_rand_xp.copy_to_host(rand_xp, stream=stream)
                # device_rand_yp.copy_to_host(rand_yp, stream=stream)
                # device_rand_delta.copy_to_host(rand_delta, stream=stream)

            stream.synchronize()
            
            # If you want to get the final values of 6D phase space coordinates
            for bunch_index, bunch_ref in enumerate(beam):

                bunch_ref["x"] = x[:, bunch_index]
                bunch_ref["xp"] = xp[:, bunch_index]
                bunch_ref["y"] = y[:, bunch_index]
                bunch_ref["yp"] = yp[:, bunch_index]
                bunch_ref["tau"] = tau[:, bunch_index]
                bunch_ref["delta"] = delta[:, bunch_index]
                  
            print("gpu_beam_emitX (zeroth bunch, first turn): " + str(beam_emitX[0, 0]))
            print("gpu_beam_emitX (last bunch, first turn): " + str(beam_emitX[0, num_bunch-1]))
            print("gpu_beam_emitY (zeroth bunch, first turn): " + str(beam_emitY[0, 0]))
            print("gpu_beam_emitY (last bunch, first turn): " + str(beam_emitY[0, num_bunch-1]))
            print("gpu_beam_emitS (zeroth bunch, first turn): " + str(beam_emitS[0, 0]))
            print("gpu_beam_emitS (last bunch, first turn): " + str(beam_emitS[0, num_bunch-1]))
            print("gpu_bunch_length (zeroth bunch, first turn): " + str(bunch_length[0, 0]))
            print("gpu_bunch_length (last bunch, first turn): " + str(bunch_length[0, num_bunch-1]))

            print("gpu_beam_emitX (zeroth bunch, last turn): " + str(beam_emitX[turns-1, 0]))
            print("gpu_beam_emitX (last bunch, last turn): " + str(beam_emitX[turns-1, num_bunch-1]))
            print("gpu_beam_emitY (zeroth bunch, last turn): " + str(beam_emitY[turns-1, 0]))
            print("gpu_beam_emitY (last bunch, last turn): " + str(beam_emitY[turns-1, num_bunch-1]))
            print("gpu_beam_emitS (zeroth bunch, last turn): " + str(beam_emitS[turns-1, 0]))
            print("gpu_beam_emitS (last bunch, last turn): " + str(beam_emitS[turns-1, num_bunch-1]))
            print("gpu_bunch_length (zeroth bunch, last turn): " + str(bunch_length[turns-1, 0]))
            print("gpu_bunch_length (last bunch, last turn): " + str(bunch_length[turns-1, num_bunch-1]))

            print("profile (first bunch, last turn): " + str(profile[:, 0]))
            print(profile[:, 0].sum())
            print(f"dip_x: \n {dip_x[:, 0]}")
            print(f"dip_y: \n {dip_y[:, 0]}")
            # print(f"dip_x: \n {dip_x[:, num_bunch-1]}")
            # print(f"dip_y: \n {dip_y[:, num_bunch-1]}")
            
            print(f"wp_tau: \n {tau[:, 0]}")
            # print(f"wp_tau: \n {wp_tau[:, num_bunch-1]}")
            print(f"wp_x: \n {wp_x[:, 0]}")
            # print(f"wp_x: \n {wp_x[:, num_bunch-1]}")
            print(f"wp_y: \n {wp_y[:, 0]}")
            # print(f"wp_y: \n {wp_y[:, num_bunch-1]}")

            # print(bin_tau)
            
            os.chdir("/home/alphaover2pi/projects/mbtrack2-cuda/")

            # filename_t = "t.bin"
            # filename_wp_tau = "wp_tau.bin"
            # filename_wp_x = "wp_x.bin"
            # filename_wp_y = "wp_y.bin"
            # with open(filename_t, "wb") as file:
            #     pickle.dump(t[:, 0], file)
            # with open(filename_wp_tau, "wb") as file:
            #     pickle.dump(wp_tau[:, 0], file)
            # with open(filename_wp_x, "wb") as file:
            #     pickle.dump(wp_x[:, 0], file)
            # with open(filename_wp_y, "wb") as file:
            #     pickle.dump(wp_y[:, 0], file)
            
            filename_tau_gpu = "tau_gpu.bin"
            filename_wp_tau_interp = "wp_tau_interp.bin"
            filename_wp_x_interp = "wp_x_interp.bin"
            filename_wp_y_interp = "wp_y_interp.bin"
            with open(filename_tau_gpu, "wb") as file:
                pickle.dump(tau[:, 0], file)
            with open(filename_wp_tau_interp, "wb") as file:
                pickle.dump(wp_tau_interp[:, 0], file)
            with open(filename_wp_x_interp, "wb") as file:
                pickle.dump(wp_x_interp[:, 0], file)
            with open(filename_wp_y_interp, "wb") as file:
                pickle.dump(wp_y_interp[:, 0], file)

            print(axis_min_tau)
            print(2*half_d_bin_tau)
            # print(t[:, 0])
            
            # plt.plot(t, dip_x)
            # plt.title("Dipole Moment x", fontsize=15)
            # plt.legend(["dip_x"], fontsize=15)
            # plt.show()

            # plt.plot(t, dip_y)
            # plt.title("Dipole Moment y", fontsize=15)
            # plt.legend(["dip_y"], fontsize=15)
            # plt.show()

            # plt.plot(t, wp_tau)
            # plt.title("Wake potential", fontsize=15)
            # plt.legend(["Wp_tau"], fontsize=15)
            # plt.show()

            # plt.plot(t, wp_x)
            # plt.title("Wake potential", fontsize=15)
            # plt.legend(["Wp_x"], fontsize=15)
            # plt.show()

            # plt.plot(t, wp_y)
            # plt.title("Wake potential", fontsize=15)
            # plt.legend(["Wp_y"], fontsize=15)
            # plt.show()

        else:
            raise ValueError("To perform GPU calculations, CUDA_PARALLEL must be enabled in the mybeam.init_beam.")