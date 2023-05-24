# -*- coding: utf-8 -*-
"""
This module defines the most basic elements for tracking, including Element,
an abstract base class which is to be used as mother class to every elements
included in the tracking.
"""
import numpy as np
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
        @cuda.jit
        def track_kernel(delta, tau, U0, E0, ac, T0, turns):

            i, j = cuda.grid(2)

            if False:
            # shared memory test
                local_i = cuda.threadIdx.x
                local_j = cuda.threadIdx.y

                delta_s = cuda.shared.array(threadperblock, numba.float32)

                for _ in range(turns):
                #Calculation of longitudinal  
                    delta_s[local_j, local_i] = delta[j, i] - U0 / E0
                    cuda.syncthreads()
                    tau[j, i] = tau[j, i] + ac * T0 * delta_s[local_j, local_i]

                    delta[j, i] = delta_s[local_j, local_i]
            else:
            # global memory
                for _ in range(turns):
                    delta[j, i] = delta[j, i] - U0 / E0
                    tau[j, i] = tau[j, i] + ac * T0 * delta[j, i]

        if isinstance(bunch, Beam):
            beam = bunch
            num_bunch = beam.__len__()
            num_particle = beam[0].mp_number
            delta = np.zeros((num_particle, num_bunch))
            tau = np.zeros((num_particle, num_bunch))

            for bunch_index, bunch_ref in enumerate(beam):
                delta[:, bunch_index] = bunch_ref['delta']
                tau[:, bunch_index] = bunch_ref['tau']
            
            threadperblock_x = 8 
            threadperblock_y = 8 
            threadperblock = (threadperblock_x, threadperblock_y) 
            blockpergrid = (num_bunch // threadperblock_x + 1, num_particle // threadperblock_y + 1)

            # Calculation in GPU 
            track_kernel[blockpergrid, threadperblock](delta, tau, self.ring.U0, self.ring.E0, self.ring.ac, self.ring.T0, turns)

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
    def track(self, bunch, turns, culm, cutm, cusr, curfc):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """
        @cuda.jit
        def track_kernel(x, xp, y, yp, tau, delta, rand_delta, rand_xp, rand_yp, num_particle,
                         num_bunch, num_p_x_b, U0, E0, ac, T0, sigma_delta, omega1, sigma_xp, sigma_yp,
                         tau_h, tau_v, tau_l, rng_states_delta, rng_states_xp, rng_states_yp,
                         m, Vc, theta, phase_advance_x_s, phase_advance_y_s, tune_x, tune_y, chro_x, chro_y,
                         pi, matrix_00, matrix_01, matrix_10, matrix_11,
                         matrix_33, matrix_34, matrix_43, matrix_44, alpha_x,
                         alpha_y, beta_x, beta_y, gamma_x, gamma_y, 
                         turns, track_error, culm, cutm, cusr, curfc):

            i, j = cuda.grid(2)

            if cutm:

                local_i = cuda.threadIdx.x
                local_j = cuda.threadIdx.y

                phase_advance_x_s = cuda.shared.array(threadperblock, numba.float32)
                phase_advance_y_s = cuda.shared.array(threadperblock, numba.float32)
                x_s = cuda.shared.array(threadperblock, numba.float32)
                xp_s = cuda.shared.array(threadperblock, numba.float32)
                y_s = cuda.shared.array(threadperblock, numba.float32)
                yp_s = cuda.shared.array(threadperblock, numba.float32)

            for _ in range(turns):

                # Longitudinal Map
                if culm:
                   delta[j, i] = delta[j, i] - U0 / E0
                   tau[j, i] = tau[j, i] + ac * T0 * delta[j, i]

                # Transverse Map
                # adts effects are ignored.
                if cutm:
                   phase_advance_x_s[local_j, local_i] = 2 * pi * (tune_x + chro_x * delta[j, i])
                   phase_advance_y_s[local_j, local_i] = 2 * pi * (tune_y + chro_y * delta[j, i])
                   cuda.syncthreads()
                   
                #    # Horizontal
                   matrix_00[j, i] = math.cos(phase_advance_x_s[local_j, local_i]) + alpha_x * math.sin(phase_advance_x_s[local_j, local_i])
                   matrix_01[j, i] = beta_x * math.sin(phase_advance_x_s[local_j, local_i])
                #    matrix_02[j, i] = dispersion_x
                   matrix_10[j, i] = -1 * gamma_x * math.sin(phase_advance_x_s[local_j, local_i])
                   matrix_11[j, i] = math.cos(phase_advance_x_s[local_j, local_i]) - alpha_x * math.sin(phase_advance_x_s[local_j, local_i])
                #    matrix_12[j, i] = dispersion_xp
                #    matrix_22[j, i] = 1

                #    # Vertical
                   matrix_33[j, i] = math.cos(phase_advance_y_s[local_j, local_i]) + alpha_y * math.sin(phase_advance_y_s[local_j, local_i])
                   matrix_34[j, i] = beta_y * math.sin(phase_advance_y_s[local_j, local_i])
                #    matrix_35[j, i] = dispersion_y
                   matrix_43[j, i] = -1 * gamma_y * math.sin(phase_advance_y_s[local_j, local_i])
                   matrix_44[j, i] = math.cos(phase_advance_y_s[local_j, local_i]) - alpha_y * math.sin(phase_advance_y_s[local_j, local_i])
                #    matrix_45[j, i] = dispersion_yp
                #    matrix_55[j, i] = 1

                   x_s[local_j, local_i] = matrix_00[j, i] * x[j, i] + matrix_01[j, i] * xp[j, i] #+ matrix_02[j, i] * delta[j, i]
                   xp_s[local_j, local_i] = matrix_10[j, i] * x[j, i] + matrix_11[j, i] * xp[j, i] #+ matrix_12[j, i] * delta[j, i]
                   y_s[local_j, local_i] =  matrix_33[j, i] * y[j, i] + matrix_34[j, i] * yp[j, i] #+ matrix_35[j, i] * delta[j, i]
                   yp_s[local_j, local_i] = matrix_43[j, i] * y[j, i] + matrix_44[j, i] * yp[j, i] #+ matrix_45[j, i] * delta[j, i]
                   cuda.syncthreads()

                   x[j, i] = x_s[local_j, local_i]
                   xp[j, i] = xp_s[local_j, local_i]
                   y[j, i] = y_s[local_j, local_i]
                   yp[j, i] = yp_s[local_j, local_i]

                # Synchrotron Radiation
                if cusr:
                   if j < num_particle and i < num_bunch:
                       idx_dxpyp = i * num_particle + j
                       if idx_dxpyp < num_p_x_b:
                          rand_delta[j, i] = xoroshiro128p_normal_float32(rng_states_delta, idx_dxpyp)
                          delta[j, i] = ((1 - 2*T0/tau_l)*delta[j, i] +
                               2*sigma_delta*(T0/tau_l)**0.5*rand_delta[j, i])
                   
                          rand_xp[j, i] = xoroshiro128p_normal_float32(rng_states_xp, idx_dxpyp)
                          xp[j, i] = ((1 - 2*T0/tau_h)*xp[j, i] +
                               2*sigma_xp*(T0/tau_h)**0.5*rand_xp[j, i])
                   
                          rand_yp[j, i] = xoroshiro128p_normal_float32(rng_states_yp, idx_dxpyp)
                          yp[j, i] = ((1 - 2*T0/tau_v)*yp[j, i] +
                               2*sigma_yp*(T0/tau_v)**0.5*rand_yp[j, i])
                          
                # RF Cavity
                if curfc:
                   delta[j, i] = delta[j, i] + Vc / E0 * math.cos(
                        m * omega1 * tau[j, i] + theta )

                if (culm == False) and (cutm == False) and (cusr == False) and (curfc == False):
                   track_error[0] = 1

        if isinstance(bunch, Beam):
            beam = bunch
            num_bunch = beam.__len__()
            num_particle = beam[0].mp_number
            x = np.zeros((num_particle, num_bunch), dtype='f')
            xp = np.zeros((num_particle, num_bunch), dtype='f')
            y = np.zeros((num_particle, num_bunch), dtype='f')
            yp = np.zeros((num_particle, num_bunch), dtype='f')
            tau = np.zeros((num_particle, num_bunch), dtype='f')
            delta = np.zeros((num_particle, num_bunch), dtype='f')
            rand_delta = np.zeros((num_particle, num_bunch), dtype='f')
            rand_xp = np.zeros((num_particle, num_bunch), dtype='f')
            rand_yp = np.zeros((num_particle, num_bunch), dtype='f')
            phase_advance_x_s = np.zeros((num_particle, num_bunch), dtype='f')
            phase_advance_y_s = np.zeros((num_particle, num_bunch), dtype='f')

            matrix_00 = np.zeros((num_particle, num_bunch), dtype='f')
            matrix_01 = np.zeros((num_particle, num_bunch), dtype='f')
            # matrix_02 = np.zeros((num_particle, num_bunch), dtype='f')
            matrix_10 = np.zeros((num_particle, num_bunch), dtype='f')
            matrix_11 = np.zeros((num_particle, num_bunch), dtype='f')
            # matrix_12 = np.zeros((num_particle, num_bunch), dtype='f')
            # matrix_22 = np.zeros((num_particle, num_bunch), dtype='f')
            matrix_33 = np.zeros((num_particle, num_bunch), dtype='f')
            matrix_34 = np.zeros((num_particle, num_bunch), dtype='f')
            # matrix_35 = np.zeros((num_particle, num_bunch), dtype='f')
            matrix_43 = np.zeros((num_particle, num_bunch), dtype='f')
            matrix_44 = np.zeros((num_particle, num_bunch), dtype='f')
            # matrix_45 = np.zeros((num_particle, num_bunch), dtype='f')
            # matrix_55 = np.zeros((num_particle, num_bunch), dtype='f')

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
            # dispersion_x = self.dispersion[0]
            # dispersion_xp = self.dispersion[1]
            # dispersion_y = self.dispersion[2]
            # dispersion_yp = self.dispersion[3]

            pi = math.pi

            track_error = np.zeros(1, dtype='f')

            for bunch_index, bunch_ref in enumerate(beam):
                x[:, bunch_index] = bunch_ref['x']
                xp[:, bunch_index] = bunch_ref['xp']
                y[:, bunch_index] = bunch_ref['y']
                yp[:, bunch_index] = bunch_ref['yp']
                tau[:, bunch_index] = bunch_ref['tau']
                delta[:, bunch_index] = bunch_ref['delta']
            
            threadperblock_x = 8 
            threadperblock_y = 8 
            threadperblock = (threadperblock_x, threadperblock_y) 
            blockpergrid = (num_bunch // threadperblock_x + 1, num_particle // threadperblock_y + 1)

            num_p_x_b = num_particle * num_bunch

            rng_states_delta = create_xoroshiro128p_states(num_p_x_b, seed=1) #seed=time.time()
            rng_states_xp = create_xoroshiro128p_states(num_p_x_b, seed=1)
            rng_states_yp = create_xoroshiro128p_states(num_p_x_b, seed=1)

            # Calculation in GPU 
            track_kernel[blockpergrid, threadperblock](x, xp, y, yp, tau, delta, rand_delta, rand_xp, rand_yp, num_particle, num_bunch, num_p_x_b,
                                                       self.ring.U0, self.ring.E0, self.ring.ac, self.ring.T0, self.ring.sigma_delta, self.ring.omega1,
                                                       sigma_xp, sigma_yp, tau_h, tau_v, tau_l, rng_states_delta, rng_states_xp, rng_states_yp, self.m,
                                                       self.Vc, self.theta, phase_advance_x_s, phase_advance_y_s, tune_x, tune_y, chro_x, chro_y, pi,
                                                       matrix_00, matrix_01, matrix_10, matrix_11, matrix_33, matrix_34,
                                                       matrix_43, matrix_44, alpha_x, alpha_y, beta_x, beta_y, gamma_x,
                                                       gamma_y, turns, track_error, culm, cutm, cusr, curfc)

            if track_error[0] == 1:
                raise ValueError("There is nothing to track.")

            for bunch_index, bunch_ref in enumerate(beam):
                bunch_ref['x'] = x[:, bunch_index]
                bunch_ref['xp'] = xp[:, bunch_index]
                bunch_ref['y'] = y[:, bunch_index]
                bunch_ref['yp'] = yp[:, bunch_index]
                bunch_ref['tau'] = tau[:, bunch_index]
                bunch_ref['delta'] = delta[:, bunch_index]

            # matrix[0, 0, :, :] = matrix_cuda[0, :, :]
            # matrix[0, 1, :, :] = matrix_cuda[1, :, :]
            # matrix[0, 2, :, :] = matrix_cuda[2, :, :]
            # matrix[1, 0, :, :] = matrix_cuda[3, :, :]
            # matrix[1, 1, :, :] = matrix_cuda[4, :, :]
            # matrix[1, 2, :, :] = matrix_cuda[5, :, :]
            # matrix[2, 2, :, :] = matrix_cuda[6, :, :]
            # matrix[3, 3, :, :] = matrix_cuda[7, :, :]
            # matrix[3, 4, :, :] = matrix_cuda[8, :, :]
            # matrix[3, 5, :, :] = matrix_cuda[9, :, :]
            # matrix[4, 3, :, :] = matrix_cuda[10, :, :]
            # matrix[4, 4, :, :] = matrix_cuda[11, :, :]
            # matrix[4, 5, :, :] = matrix_cuda[12, :, :]
            # matrix[5, 5, :, :] = matrix_cuda[13, :, :]

        else:
            raise ValueError("To perform GPU calculations, CUDA_PARALLEL must be enabled in the mybeam.init_beam.")