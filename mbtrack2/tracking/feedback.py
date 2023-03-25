# -*- coding: utf-8 -*-
"""
This module defines both exponential and FIR based bunch by bunch damper 
feedback for tracking.
"""
import numpy as np 
import matplotlib.pyplot as plt
from mbtrack2.tracking import Element, Beam, Bunch

class ExponentialDamper(Element): 
    """ 
    Simple bunch by bunch damper feedback system based on exponential damping.
    
    Parameters
    ----------
    ring : Synchrotron object
        Synchrotron to use.
    plane : {"x","y","s"}
        Allow to choose on which plane the damper is active.
    damping_time : float
        Damping time in [s].
    phase_diff : array of shape (3,)
        Phase difference between the damper and the position monitor in [rad].
        
    """
    def __init__(self, ring, plane, damping_time, phase_diff):
        self.ring = ring
        self.damping_time = damping_time
        self.phase_diff = phase_diff
        self.plane = plane
        if self.plane == "x":
            self.action = "xp"
            self.damp_idx = 0
            self.mean_idx = 1
        elif self.plane == "y":
            self.action = "yp"
            self.damp_idx = 1
            self.mean_idx = 3
        elif self.plane == "s":
            self.action = "delta"
            self.damp_idx = 2
            self.mean_idx = 5
        else:
            raise ValueError(f"plane should be x, y or s, not {self.plane}")
            
    @Element.parallel 
    def track(self, bunch):
        """
        Tracking method for the feedback system
        No bunch to bunch interaction, so written for Bunch object and
        @Element.parallel is used to handle Beam object.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """
        bunch[self.action] -= (2*self.ring.T0/
                               self.damping_time[self.damp_idx]*
                               np.sin(self.phase_diff)*
                               bunch.mean[self.mean_idx])

class FIRDamper(Element): 
    """ 
    Bunch by bunch damper feedback system based on FIR filters.
    
    FIR computation is based on [1].
    
    Parameters
    ----------
    ring : Synchrotron object
        Synchrotron to use.
    plane : {"x","y","s"}
        Allow to choose on which plane the damper is active.
    tune : float
        Reference (betatron or synchrotron) tune for which the damper system 
        is set.
    turn_delay : int
        Number of turn delay before the kick is applied.
    tap_number : int
        Number of tap for the FIR filter.
    gain : float
        Gain of the FIR filter.
    phase : float
        Phase of the FIR filter in [degree].
    meas_error : float, optional
        RMS measurement error applied to the computed mean position.
        Unit is [m] if the plane is "x" or "y" and [s] if the plane is "s".
        The default is None.
    max_kick : float, optional
        Maximum kick strength limitation.
        Unit is [rad] if the plane is "x" or "y" and no unit (delta) if the 
        plane is "s".
        The default is None.

    Attributes
    ----------
    pos : array
        Stored beam postions.
    kick : array
        Stored damper kicks.
    coef : array
        Coefficients of the FIR filter.
         
    Methods
    -------
    get_fir(tap_number, tune, phase, turn_delay, gain)
        Initialize the FIR filter and return an array containing the FIR 
        coefficients.
    plot_fir()
        Plot the gain and the phase of the FIR filter.
    track(beam_or_bunch)
        Tracking method.
      
    References
    ----------
    [1] T.Nakamura, S.Daté, K. Kobayashi, T. Ohshima. Proceedings of EPAC 
    2004. Transverse bunch by bunch feedback system for the Spring-8 
    storage ring.
    """
    
    def __init__(self, ring, plane, tune, turn_delay, tap_number, gain, phase, 
                 bpm_error=None, max_kick=None):
        
        self.ring = ring
        self.tune = tune
        self.turn_delay = turn_delay
        self.tap_number = tap_number
        self.gain = gain
        self.phase = phase
        self.bpm_error = bpm_error
        self.max_kick = max_kick
        self.plane = plane
        
        if self.plane == "x":
            self.action = "xp"
            self.damp_idx = 0
            self.mean_idx = 0
        elif self.plane == "y":
            self.action = "yp"
            self.damp_idx = 1
            self.mean_idx = 2
        elif self.plane == "s":
            self.action = "delta"
            self.damp_idx = 2
            self.mean_idx = 4
            
        self.beam_no_mpi = False
        
        self.pos = np.zeros((self.tap_number,1))
        self.kick = np.zeros((self.turn_delay+1,1))
        self.coef = self.get_fir(self.tap_number, self.tune, self.phase, 
                                   self.turn_delay, self.gain)
        
    def get_fir(self, tap_number, tune, phase, turn_delay, gain):        
        """
        Compute the FIR coefficients.
        
        FIR computation is based on [1].
        
        Returns
        -------
        FIR_coef : array
            Array containing the FIR coefficients.
        """
        it = np.zeros((tap_number,))
        CC = np.zeros((5, tap_number,))
        zeta = (phase*2*np.pi)/360
        for k in range(tap_number):
            it[k] = (-k - turn_delay)
        
        phi = 2*np.pi*tune
        cs = np.cos(phi*it)
        sn = np.sin(phi*it)
        
        CC[0][:] = 1
        CC[1][:] = cs
        CC[2][:] = sn
        CC[3][:] = it*sn
        CC[4][:] = it*cs
        
        TCC = np.transpose(CC)
        W = np.linalg.inv(CC.dot(TCC))
        D = W.dot(CC)
        
        FIR_coef = gain*(D[1][:]*np.cos(zeta) + D[2][:]*np.sin(zeta))
        return FIR_coef
    
    def plot_fir(self):
        """
        Plot the gain and the phase of the FIR filter.
        
        Returns
        -------
        fig : Figure
            Plot of the gain and phase.
            
        """
        tune = np.arange(0, 1, 0.0001)
            
        H_FIR = 0
        for k in range(len(self.coef)):
            H_FIR += self.coef[k]*np.exp(-1j*2*np.pi*(k)*tune)
        latency = np.exp(-1j*2*np.pi*tune*self.turn_delay)
        H_tot = H_FIR * latency
        
        gain = np.abs(H_tot)
        phase = np.angle(H_tot, deg = True)
        
        fig, [ax1, ax2] = plt.subplots(2,1)
        ax1.plot(tune, gain)
        ax1.set_ylabel("Gain")
        
        ax2.plot(tune, phase)
        ax2.set_xlabel("Tune")
        ax2.set_ylabel("Phase in degree")
        
        return fig
         
    def track(self, beam_or_bunch):
        """
        Tracking method.

        Parameters
        ----------
        beam_or_bunch : Beam or Bunch
            Data to track.
            
        """
        if isinstance(beam_or_bunch, Bunch):
            self.track_sb(beam_or_bunch)
        elif isinstance(beam_or_bunch, Beam):
            beam = beam_or_bunch
            if (beam.mpi_switch == True):
                self.track_sb(beam[beam.mpi.bunch_num])
            else:
                if self.beam_no_mpi is False:
                    self.init_beam_no_mpi(beam)
                for i, bunch in enumerate(beam.not_empty):
                    self.track_sb(bunch, i)
        else:
            TypeError("beam_or_bunch must be a Beam or Bunch")
            
    def init_beam_no_mpi(self, beam):
        """
        Change array sizes if Beam is used without mpi.

        Parameters
        ----------
        beam : Beam
            Beam to track.

        """
        n_bunch = len(beam)
        self.pos = np.zeros((self.tap_number, n_bunch))
        self.kick = np.zeros((self.turn_delay+1, n_bunch))
        self.beam_no_mpi = True
        
    def track_sb(self, bunch, bunch_number=0):
        """
        Core of the tracking method.

        Parameters
        ----------
        bunch : Bunch
            Bunch to track.
        bunch_number : int, optional
            Number of bunch in beam.not_empty. 
            The default is 0.
            
        """
        self.pos[0, bunch_number] = bunch.mean[self.mean_idx]
        if self.bpm_error is not None:
            self.pos[0, bunch_number] += np.random.normal(0, self.bpm_error)
            
        kick = 0
        for k in range(self.tap_number):
            kick += self.coef[k]*self.pos[k, bunch_number]
            
        if self.max_kick is not None:
            if kick > self.max_kick:
                kick = self.max_kick
            elif kick < -1*self.max_kick:
                kick = -1*self.max_kick
            
        self.kick[-1, bunch_number] = kick
        bunch[self.action] += self.kick[0, bunch_number]
        
        self.pos[:, bunch_number] = np.roll(self.pos[:, bunch_number], 1)
        self.kick[:, bunch_number] = np.roll(self.kick[:, bunch_number], -1)