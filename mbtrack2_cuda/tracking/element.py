# -*- coding: utf-8 -*-
"""
This module defines the most basic elements for tracking, including Element,
an abstract base class which is to be used as mother class to every elements
included in the tracking.
"""
import numpy as np
import time
import math
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
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
    def track(self, bunch, turns):
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
            phase_advance_x_s = 2*np.pi * (self.ring.tune[0] + 
                                         self.ring.chro[0]*bunch["delta"])
            phase_advance_y_s = 2*np.pi * (self.ring.tune[1] + 
                                         self.ring.chro[1]*bunch["delta"])
        else:
            phase_advance_x_s = 2*np.pi * (self.ring.tune[0] + 
                                         self.ring.chro[0]*bunch["delta"] + 
                                         self.adts_poly[0](bunch['x']) + 
                                         self.adts_poly[2](bunch['y']))
            phase_advance_y_s = 2*np.pi * (self.ring.tune[1] + 
                                         self.ring.chro[1]*bunch["delta"] +
                                         self.adts_poly[1](bunch['x']) + 
                                         self.adts_poly[3](bunch['y']))
        
        # 6x6 matrix corresponding to (x, xp, delta, y, yp, delta)
        matrix = np.zeros((6,6,len(bunch)))
        
        # Horizontal
        matrix[0,0,:] = np.cos(phase_advance_x_s) + self.alpha[0]*np.sin(phase_advance_x_s)
        matrix[0,1,:] = self.beta[0]*np.sin(phase_advance_x_s)
        matrix[0,2,:] = self.dispersion[0]
        matrix[1,0,:] = -1*self.gamma[0]*np.sin(phase_advance_x_s)
        matrix[1,1,:] = np.cos(phase_advance_x_s) - self.alpha[0]*np.sin(phase_advance_x_s)
        matrix[1,2,:] = self.dispersion[1]
        matrix[2,2,:] = 1
        
        # Vertical
        matrix[3,3,:] = np.cos(phase_advance_y_s) + self.alpha[1]*np.sin(phase_advance_y_s)
        matrix[3,4,:] = self.beta[1]*np.sin(phase_advance_y_s)
        matrix[3,5,:] = self.dispersion[2]
        matrix[4,3,:] = -1*self.gamma[1]*np.sin(phase_advance_y_s)
        matrix[4,4,:] = np.cos(phase_advance_y_s) - self.alpha[1]*np.sin(phase_advance_y_s)
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

class CUDAMap(Element):
    """
    Longitudinal Map, Transverse Map, Synchrotron Radiation, RF Cavity, Resistive Wall for GPU calculations.
    
    Parameters
    ----------
    ring : Synchrotron object
    """

    def __init__(self, ring, m, Vc, theta):
        self.ring = ring
        self.m = m 
        self.Vc = Vc
        self.theta = theta
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
    def track(self, bunch, turns, culm, cutm, cusr, curfc, cubm):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """
        @cuda.jit
        def track_kernel(x_read, xp_read, y_read, yp_read, tau_read, delta_read,
                         num_particle, num_bunch, U0, E0, ac, T0, sigma_delta, omega1, sigma_xp, sigma_yp,
                         tau_h, tau_v, tau_l, rng_states, dispersion_x, dispersion_xp, dispersion_y, dispersion_yp,
                         m, Vc, theta, tune_x, tune_y, chro_x, chro_y,
                         pi, alpha_x, alpha_y, beta_x, beta_y, gamma_x, gamma_y,
                         turns_sum_x_squared, turns_sum_xp_squared, turns_sum_x_xp, turns_sum_y_squared,
                         turns_sum_yp_squared, turns_sum_y_yp, turns_sum_tau_squared, turns_sum_delta_squared,
                         turns_sum_tau_delta, turns, track_error, culm, cutm, cusr, curfc, cubm,
                         x_write, xp_write, y_write, yp_write, tau_write, delta_write):
            
            i, j = cuda.grid(2)

            local_i = cuda.threadIdx.x
            local_j = cuda.threadIdx.y

            # No Selection Error 
            if (culm == False) and (cutm == False) and (cusr == False) and (curfc == False):
               track_error[0] = 1
            
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

            sum_x_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_xp_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_x_xp_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_y_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_yp_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_y_yp_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_tau_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_delta_squared_shared = cuda.shared.array(threadperblock, numba.float32)
            sum_tau_delta_shared = cuda.shared.array(threadperblock, numba.float32)

            rand_shared = cuda.shared.array(threadperblock, numba.float32)

            x_shared[local_j, local_i] = x_read[j, i]
            xp_shared[local_j, local_i] = xp_read[j, i]
            y_shared[local_j, local_i] = y_read[j, i]
            yp_shared[local_j, local_i] = yp_read[j, i]
            tau_shared[local_j, local_i] = tau_read[j, i]
            delta_shared[local_j, local_i] = delta_read[j, i]

            cuda.syncthreads()

            for k in range(turns):

                # Longitudinal Map
                if culm:
                   
                   delta_shared[local_j, local_i] -= U0 / E0
                   cuda.syncthreads()
                   tau_shared[local_j, local_i] += ac * T0 * delta_shared[local_j, local_i]
                   cuda.syncthreads()

                   delta_write[j, i] = delta_shared[local_j, local_i]
                   tau_write[j, i] = tau_shared[local_j, local_i]

                # Transverse Map
                # adts effects are ignored. (Future work)
                if cutm:
                   
                   x_shared_f[local_j, local_i] = ( ( math.cos(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) +
                           alpha_x * math.sin(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) ) * x_shared[local_j, local_i] + 
                           ( beta_x * math.sin(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) ) * xp_shared[local_j, local_i] +
                           dispersion_x * delta_shared[local_j, local_i] )

                   xp_shared_f[local_j, local_i] = ( ( -1 * gamma_x * math.sin(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) ) * x_shared[local_j, local_i] +
                           ( math.cos(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) -
                           alpha_x * math.sin(2 * pi * (tune_x + chro_x * delta_shared[local_j, local_i])) ) * xp_shared[local_j, local_i] +
                           dispersion_xp * delta_shared[local_j, local_i] )
                   
                   cuda.syncthreads()

                   x_write[j, i] = x_shared_f[local_j, local_i]
                   xp_write[j, i] = xp_shared_f[local_j, local_i]

                   x_shared[local_j, local_i] = x_shared_f[local_j, local_i]
                   xp_shared[local_j, local_i] = xp_shared_f[local_j, local_i]

                   y_shared_f[local_j, local_i] = ( ( math.cos(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) +
                           alpha_y * math.sin(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) ) * y_shared[local_j, local_i] + 
                           ( beta_y * math.sin(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) ) * yp_shared[local_j, local_i] +
                           dispersion_y * delta_shared[local_j, local_i] )

                   yp_shared_f[local_j, local_i] = ( ( -1 * gamma_y * math.sin(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) ) * y_shared[local_j, local_i] + 
                           ( math.cos(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) -
                           alpha_y * math.sin(2 * pi * (tune_y + chro_y * delta_shared[local_j, local_i])) ) * yp_shared[local_j, local_i] +
                           dispersion_yp * delta_shared[local_j, local_i] )
                   
                   cuda.syncthreads()

                   y_write[j, i] = y_shared_f[local_j, local_i]
                   yp_write[j, i] = yp_shared_f[local_j, local_i]
                   
                   y_shared[local_j, local_i] = y_shared_f[local_j, local_i]
                   yp_shared[local_j, local_i] = yp_shared_f[local_j, local_i]

                   cuda.syncthreads()

                # Synchrotron Radiation
                if cusr:
                   #rand_xp
                   if local_j < num_particle and local_i < num_bunch:
                        rand_shared[local_j, local_i] = xoroshiro128p_normal_float32(rng_states, local_j + num_particle * local_i)
                   cuda.syncthreads()
                   xp_shared[local_j, local_i] = ( (1 - 2*T0/tau_h) * xp_shared[local_j, local_i] +
                        2*sigma_xp*(T0/tau_h)**0.5 * rand_shared[local_j, local_i] )

                   #rand_yp
                   if local_j < num_particle and local_i < num_bunch:
                        rand_shared[local_j, local_i] = xoroshiro128p_normal_float32(rng_states, local_j + num_particle * local_i)
                   cuda.syncthreads()
                   yp_shared[local_j, local_i] = ( (1 - 2*T0/tau_v) * yp_shared[local_j, local_i] +
                        2*sigma_yp*(T0/tau_v)**0.5 * rand_shared[local_j, local_i] )
                          
                   #rand_delta
                   if local_j < num_particle and local_i < num_bunch:
                        rand_shared[local_j, local_i] = xoroshiro128p_normal_float32(rng_states, local_j + num_particle * local_i)
                   cuda.syncthreads()
                   delta_shared[local_j, local_i] = ( (1 - 2*T0/tau_l) * delta_shared[local_j, local_i] +
                        2*sigma_delta*(T0/tau_l)**0.5 * rand_shared[local_j, local_i] )
                          
                   cuda.syncthreads()

                   xp_write[j, i] = xp_shared[local_j, local_i]
                   yp_write[j, i] = yp_shared[local_j, local_i]
                   delta_write[j, i] = delta_shared[local_j, local_i]

                # RF Cavity
                if curfc:
                   delta_shared[local_j, local_i] += Vc / E0 * math.cos(
                        m * omega1 * tau_shared[local_j, local_i] + theta )
                   cuda.syncthreads()

                   delta_write[j, i] = delta_shared[local_j, local_i]

                # Beam Monitor
                if cubm:
                   sum_x_squared_shared[local_j, local_i] = x_shared[local_j, local_i]**2
                   sum_xp_squared_shared[local_j, local_i] = xp_shared[local_j, local_i]**2
                   sum_x_xp_shared[local_j, local_i] = x_shared[local_j, local_i] * xp_shared[local_j, local_i] 
                   sum_y_squared_shared[local_j, local_i] = y_shared[local_j, local_i]**2
                   sum_yp_squared_shared[local_j, local_i] = yp_shared[local_j, local_i]**2
                   sum_y_yp_shared[local_j, local_i] = y_shared[local_j, local_i] * yp_shared[local_j, local_i]
                   sum_tau_squared_shared[local_j, local_i] = tau_shared[local_j, local_i]**2
                   sum_delta_squared_shared[local_j, local_i] = delta_shared[local_j, local_i]**2
                   sum_tau_delta_shared[local_j, local_i] = tau_shared[local_j, local_i] * delta_shared[local_j, local_i]
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
                     cuda.syncthreads()
                     s >>= 1
                
                   if local_j == 0 and i < num_bunch:
                        turns_sum_x_squared[cuda.blockIdx.y, i, k] = sum_x_squared_shared[0, local_i]
                        turns_sum_xp_squared[cuda.blockIdx.y, i, k] = sum_xp_squared_shared[0, local_i]
                        turns_sum_x_xp[cuda.blockIdx.y, i, k] = sum_x_xp_shared[0, local_i]
                        turns_sum_y_squared[cuda.blockIdx.y, i, k] = sum_y_squared_shared[0, local_i]
                        turns_sum_yp_squared[cuda.blockIdx.y, i, k] = sum_yp_squared_shared[0, local_i]
                        turns_sum_y_yp[cuda.blockIdx.y, i, k] = sum_y_yp_shared[0, local_i]
                        turns_sum_tau_squared[cuda.blockIdx.y, i, k] = sum_tau_squared_shared[0, local_i]
                        turns_sum_delta_squared[cuda.blockIdx.y, i, k] = sum_delta_squared_shared[0, local_i]
                        turns_sum_tau_delta[cuda.blockIdx.y, i, k] = sum_tau_delta_shared[0, local_i]

        if isinstance(bunch, Beam):
            beam = bunch
            num_bunch = beam.__len__()
            num_particle = beam[0].mp_number
            x_read = np.zeros((num_particle, num_bunch), dtype='f')
            xp_read = np.zeros((num_particle, num_bunch), dtype='f')
            y_read = np.zeros((num_particle, num_bunch), dtype='f')
            yp_read = np.zeros((num_particle, num_bunch), dtype='f')
            tau_read = np.zeros((num_particle, num_bunch), dtype='f')
            delta_read = np.zeros((num_particle, num_bunch), dtype='f')
            
            #If you want to get the final values of 6D phase space coordinates
            x_write = np.zeros((num_particle, num_bunch), dtype='f')
            xp_write = np.zeros((num_particle, num_bunch), dtype='f')
            y_write = np.zeros((num_particle, num_bunch), dtype='f')
            yp_write = np.zeros((num_particle, num_bunch), dtype='f')
            tau_write = np.zeros((num_particle, num_bunch), dtype='f')
            delta_write = np.zeros((num_particle, num_bunch), dtype='f')

            #If you don't want to get the final values of 6D phase space coordinates
            # x_write = 0.0
            # xp_write = 0.0
            # y_write = 0.0
            # yp_write = 0.0
            # tau_write = 0.0
            # delta_write = 0.0

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

            pi = np.pi

            track_error = np.zeros(1, dtype='f')

            for bunch_index, bunch_ref in enumerate(beam):
                x_read[:, bunch_index] = bunch_ref['x']
                xp_read[:, bunch_index] = bunch_ref['xp']
                y_read[:, bunch_index] = bunch_ref['y']
                yp_read[:, bunch_index] = bunch_ref['yp']
                tau_read[:, bunch_index] = bunch_ref['tau']
                delta_read[:, bunch_index] = bunch_ref['delta']
            
            threadperblock_x = 16
            threadperblock_y = 16
            threadperblock = (threadperblock_x, threadperblock_y)
            blockpergrid = (num_bunch // threadperblock_x + 1, num_particle // threadperblock_y + 1)

            rng_states = create_xoroshiro128p_states(num_particle*num_bunch, seed=time.time())

            turns_sum_x_squared = np.zeros((blockpergrid[1], num_bunch, turns), dtype='f')
            turns_sum_xp_squared = np.zeros((blockpergrid[1], num_bunch, turns), dtype='f')
            turns_sum_x_xp = np.zeros((blockpergrid[1], num_bunch, turns), dtype='f')
            turns_sum_y_squared = np.zeros((blockpergrid[1], num_bunch, turns), dtype='f')
            turns_sum_yp_squared = np.zeros((blockpergrid[1], num_bunch, turns), dtype='f')
            turns_sum_y_yp = np.zeros((blockpergrid[1], num_bunch, turns), dtype='f')
            turns_sum_tau_squared = np.zeros((blockpergrid[1], num_bunch, turns), dtype='f')
            turns_sum_delta_squared = np.zeros((blockpergrid[1], num_bunch, turns), dtype='f')
            turns_sum_tau_delta = np.zeros((blockpergrid[1], num_bunch, turns), dtype='f')

            # Calculation in GPU 

            track_kernel[blockpergrid, threadperblock](x_read, xp_read, y_read, yp_read, tau_read, delta_read, num_particle, num_bunch,
                                                       self.ring.U0, self.ring.E0, self.ring.ac, self.ring.T0, self.ring.sigma_delta, self.ring.omega1,
                                                       sigma_xp, sigma_yp, tau_h, tau_v, tau_l, rng_states, dispersion_x, dispersion_xp, dispersion_y,
                                                       dispersion_yp, self.m, self.Vc, self.theta, tune_x, tune_y, chro_x, chro_y, pi,
                                                       alpha_x, alpha_y, beta_x, beta_y, gamma_x, gamma_y,
                                                       turns_sum_x_squared, turns_sum_xp_squared, turns_sum_x_xp, turns_sum_y_squared,
                                                       turns_sum_yp_squared, turns_sum_y_yp, turns_sum_tau_squared, turns_sum_delta_squared,
                                                       turns_sum_tau_delta, turns, track_error, culm, cutm, cusr, curfc, cubm,
                                                       x_write, xp_write, y_write, yp_write, tau_write, delta_write)

            if track_error[0] == 1:
                raise ValueError("There is nothing to track.")
            
            #If you want to get the final values of 6D phase space coordinates
            for bunch_index, bunch_ref in enumerate(beam):
                bunch_ref['x'] = x_write[:, bunch_index]
                bunch_ref['xp'] = xp_write[:, bunch_index]
                bunch_ref['y'] = y_write[:, bunch_index]
                bunch_ref['yp'] = yp_write[:, bunch_index]
                bunch_ref['tau'] = tau_write[:, bunch_index]
                bunch_ref['delta'] = delta_write[:, bunch_index]

            turns_beam_emitX = ( (turns_sum_x_squared.sum(axis=0) * turns_sum_xp_squared.sum(axis=0) - turns_sum_x_xp.sum(axis=0)**2) / (num_particle**2) )**(0.5)
            turns_beam_emitY = ( (turns_sum_y_squared.sum(axis=0) * turns_sum_yp_squared.sum(axis=0) - turns_sum_y_yp.sum(axis=0)**2) / num_particle**2 )**(0.5)
            turns_beam_emitS = ( (turns_sum_tau_squared.sum(axis=0) * turns_sum_delta_squared.sum(axis=0) - turns_sum_tau_delta.sum(axis=0)**2) / num_particle**2 )**(0.5)
            
            print('gpu_beam_emitX (zeroth bunch, first turn): ' + str(turns_beam_emitX[0, 0]))
            print('gpu_beam_emitX (last bunch, first turn): ' + str(turns_beam_emitX[num_bunch-1, 0]))
            print('gpu_beam_emitY (zeroth bunch, first turn): ' + str(turns_beam_emitY[0, 0]))
            print('gpu_beam_emitY (last bunch, first turn): ' + str(turns_beam_emitY[num_bunch-1, 0]))
            print('gpu_beam_emitS (zeroth bunch, first turn): ' + str(turns_beam_emitS[0, 0]))
            print('gpu_beam_emitS (last bunch, first turn): ' + str(turns_beam_emitS[num_bunch-1, 0]))

            print('gpu_beam_emitX (zeroth bunch, last turn): ' + str(turns_beam_emitX[0, turns-1]))
            print('gpu_beam_emitX (last bunch, last turn): ' + str(turns_beam_emitX[num_bunch-1, turns-1]))
            print('gpu_beam_emitY (zeroth bunch, last turn): ' + str(turns_beam_emitY[0, turns-1]))
            print('gpu_beam_emitY (last bunch, last turn): ' + str(turns_beam_emitY[num_bunch-1, turns-1]))
            print('gpu_beam_emitS (zeroth bunch, last turn): ' + str(turns_beam_emitS[0, turns-1]))
            print('gpu_beam_emitS (last bunch, last turn): ' + str(turns_beam_emitS[num_bunch-1, turns-1]))

        else:
            raise ValueError("To perform GPU calculations, CUDA_PARALLEL must be enabled in the mybeam.init_beam.")