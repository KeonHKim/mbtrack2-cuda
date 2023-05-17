# -*- coding: utf-8 -*-
"""
This module defines the most basic elements for tracking, including Element,
an abstract base class which is to be used as mother class to every elements
included in the tracking.
"""
import numpy as np
import numba
from numba import cuda
from abc import ABCMeta, abstractmethod
from functools import wraps
from mbtrack2_cuda.tracking.particles import Beam

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
                if (beam.cuda_switch == True):
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
        @cuda.jit
        def track_kernel(delta, tau, U0, E0, ac, T0):

            i, j = cuda.grid(2)

            if False:
            # shared memory test
                local_i = cuda.threadIdx.x
                local_j = cuda.threadIdx.y

                delta_s = cuda.shared.array(threadperblock, numba.float32)

                #Calculation of longitudinal  
                delta_s[local_j, local_i] = delta[j, i] - U0 / E0
                cuda.syncthreads()
                tau[j, i] = tau[j, i] + ac * T0 * delta_s[local_j, local_i]

                delta[j, i] = delta_s[local_j, local_i]
            else:
            # global memory
                delta[j, i] = delta[j, i] - U0 / E0
                tau[j, i] = tau[j, i] + ac * T0 * delta[j, i]

        if isinstance(bunch, Beam):
            beam = bunch
            num_bunch = beam.__len__()
            num_particle = beam[0].mp_number
            delta = np.zeros((num_particle, num_bunch))
            tau = np.zeros((num_particle, num_bunch))

            for bunch_index, bunch_ref in enumerate(beam):
                delta[:,bunch_index] = bunch_ref['delta']
                tau[:,bunch_index] = bunch_ref['tau']
            
            threadperblock_x = 8 
            threadperblock_y = 8 
            threadperblock = (threadperblock_x, threadperblock_y) 
            blockpergrid = (num_bunch // threadperblock_x + 1, num_particle // threadperblock_y + 1)

            # Calculation in GPU 
            track_kernel[blockpergrid, threadperblock](delta, tau, self.ring.U0, self.ring.E0, self.ring.ac, self.ring.T0)

            for bunch_index, bunch_ref in enumerate(beam):
                bunch_ref['delta'] = delta[:, bunch_index]
                bunch_ref['tau'] = tau[:, bunch_index]

        else:
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
                 2*self.ring.sigma_delta*(self.ring.T0/self.ring.tau[2])**0.5*rand)
            
        if (self.switch[1] == True):
            rand = np.random.normal(size=len(bunch))
            bunch["xp"] = ((1 - 2*self.ring.T0/self.ring.tau[0])*bunch["xp"] +
                 2*self.ring.sigma()[1]*(self.ring.T0/self.ring.tau[0])**0.5*rand)
       
        if (self.switch[2] == True):
            rand = np.random.normal(size=len(bunch))
            bunch["yp"] = ((1 - 2*self.ring.T0/self.ring.tau[1])*bunch["yp"] +
                 2*self.ring.sigma()[3]*(self.ring.T0/self.ring.tau[1])**0.5*rand)
        
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
            phase_advance_x = 2*np.pi * (self.ring.tune[0] + 
                                         self.ring.chro[0]*bunch["delta"])
            phase_advance_y = 2*np.pi * (self.ring.tune[1] + 
                                         self.ring.chro[1]*bunch["delta"])
        else:
            phase_advance_x = 2*np.pi * (self.ring.tune[0] + 
                                         self.ring.chro[0]*bunch["delta"] + 
                                         self.adts_poly[0](bunch['x']) + 
                                         self.adts_poly[2](bunch['y']))
            phase_advance_y = 2*np.pi * (self.ring.tune[1] + 
                                         self.ring.chro[1]*bunch["delta"] +
                                         self.adts_poly[1](bunch['x']) + 
                                         self.adts_poly[3](bunch['y']))
        
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

