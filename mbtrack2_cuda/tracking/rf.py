# -*- coding: utf-8 -*-
"""
This module handles radio-frequency (RF) cavitiy elements. 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from mbtrack2_cuda.tracking.element import Element

class RFCavity(Element):
    """
    Perfect RF cavity class for main and harmonic RF cavities.
    Use cosine definition.
    
    Parameters
    ----------
    ring : Synchrotron object
    m : int
        Harmonic number of the cavity
    Vc : float
        Amplitude of cavity voltage [V]
    theta : float
        Phase of Cavity voltage
    """
    def __init__(self, ring, m, Vc, theta):
        self.ring = ring
        self.m = m 
        self.Vc = Vc
        self.theta = theta
        
    @Element.parallel    
    def track(self,bunch):
        """
        Tracking method for the element.
        No bunch to bunch interaction, so written for Bunch objects and
        @Element.parallel is used to handle Beam objects.
        
        Parameters
        ----------
        bunch : Bunch or Beam object
        """
        bunch["delta"] += self.Vc / self.ring.E0 * np.cos(
                self.m * self.ring.omega1 * bunch["tau"] + self.theta )
        
    def value(self, val):
        return self.Vc / self.ring.E0 * np.cos( 
                self.m * self.ring.omega1 * val + self.theta )
    
    
class CavityResonator():
    """Cavity resonator class for active or passive RF cavity with beam
    loading or HOM, based on [1,2].
    
    Use cosine definition.
    
    If used with mpi, beam.mpi.share_distributions must be called before the 
    track method call.
    
    Parameters
    ----------
    ring : Synchrotron object
    m : int or float
        Harmonic number of the cavity.
    Rs : float
        Shunt impedance of the cavities in [Ohm], defined as 0.5*Vc*Vc/Pc.
        If Ncav = 1, used for the total shunt impedance.
        If Ncav > 1, used for the shunt impedance per cavity.
    Q : float
        Quality factor of the cavity.
    QL : float
        Loaded quality factor of the cavity.
    detune : float
        Detuing of the cavity in [Hz], defined as (fr - m*ring.f1).
    Ncav : int, optional
        Number of cavities.
    Vc : float, optinal
        Total cavity voltage in [V].
    theta : float, optional
        Total cavity phase in [rad].
    n_bin : int, optional
        Number of bins used for the beam loading computation. 
        Only used if MPI is not used, otherwise n_bin must be specified in the 
        beam.mpi.share_distributions method.
        The default is 75.
        
    Attributes
    ----------
    beam_phasor : complex
        Beam phasor in [V].
    beam_phasor_record : array of complex
        Last beam phasor value of each bunch in [V].
    generator_phasor : complex
        Generator phasor in [V].
    cavity_phasor : complex
        Cavity phasor in [V].
    cavity_phasor_record : array of complex
        Last cavity phasor value of each bunch in [V].
    cavity_voltage : float
        Cavity total voltage in [V].
    cavity_phase : float
        Cavity total phase in [rad].
    loss_factor : float
        Cavity loss factor in [V/C].
    Rs_per_cavity : float
        Shunt impedance of a single cavity in [Ohm], defined as 0.5*Vc*Vc/Pc.
    beta : float
        Coupling coefficient of the cavity.
    fr : float
        Resonance frequency of the cavity in [Hz].
    wr : float
        Angular resonance frequency in [Hz.rad].
    psi : float
        Tuning angle in [rad].
    filling_time : float
        Cavity filling time in [s].
    Pc : float
        Power dissipated in the cavity walls in [W].
    Pg : float
        Generator power in [W].
    Vgr : float
        Generator voltage at resonance in [V].
    theta_gr : float
        Generator phase at resonance in [rad].
    Vg : float
        Generator voltage in [V].
    theta_g : float
        Generator phase in [rad].
    tracking : bool
        True if the tracking has been initialized.
    bunch_index : int
        Number of the tracked bunch in the current core.
    distance : array
        Distance between bunches.
    valid_bunch_index : array
    
    Methods
    -------
    Vbr(I0)
        Return beam voltage at resonance in [V].
    Vb(I0)
        Return beam voltage in [V].
    Pb(I0)
        Return power transmitted to the beam in [W].
    Pr(I0)
        Return power reflected back to the generator in [W].
    Z(f)
        Cavity impedance in [Ohm] for a given frequency f in [Hz].
    set_optimal_coupling(I0)
        Set coupling to optimal value.
    set_optimal_detune(I0)
        Set detuning to optimal conditions.
    set_generator(I0)
        Set generator parameters.
    plot_phasor(I0)
        Plot phasor diagram.
    is_DC_Robinson_stable(I0)
        Check DC Robinson stability.
    plot_DC_Robinson_stability()
        Plot DC Robinson stability limit.
    init_tracking(beam)
        Initialization of the tracking.
    track(beam)
        Tracking method.
    phasor_decay(time)
        Compute the beam phasor decay during a given time span.
    phasor_evol(profile, bin_length, charge_per_mp)
        Compute the beam phasor evolution during the crossing of a bunch.
    VRF(z, I0)
        Return the total RF voltage.
    dVRF(z, I0)
        Return derivative of total RF voltage.
    ddVRF(z, I0)
        Return the second derivative of total RF voltage.
    deltaVRF(z, I0)
        Return the generator voltage minus beam loading voltage.
    
    References
    ----------
    [1] Wilson, P. B. (1994). Fundamental-mode rf design in e+ e− storage ring 
    factories. In Frontiers of Particle Beams: Factories with e+ e-Rings 
    (pp. 293-311). Springer, Berlin, Heidelberg.
    
    [2] Yamamoto, Naoto, Alexis Gamelin, and Ryutaro Nagaoka. "Investigation 
    of Longitudinal Beam Dynamics With Harmonic Cavities by Using the Code 
    Mbtrack." IPAC’19, Melbourne, Australia, 2019.
    """
    def __init__(self, ring, m, Rs, Q, QL, detune, Ncav=1, Vc=0, theta=0, 
                 n_bin=75):
        self.ring = ring
        self.m = m
        self.Ncav = Ncav
        if Ncav != 1:
            self.Rs_per_cavity = Rs
        else:
            self.Rs = Rs
        self.Q = Q
        self.QL = QL
        self.detune = detune
        self.Vc = Vc
        self.theta = theta
        self.beam_phasor = np.zeros(1, dtype=complex)
        self.beam_phasor_record = np.zeros((self.ring.h), dtype=complex)
        self.tracking = False
        self.Vg = 0
        self.theta_g = 0
        self.Vgr = 0
        self.theta_gr = 0
        self.Pg = 0
        self.n_bin = int(n_bin)
        
    def init_tracking(self, beam):
        """
        Initialization of the tracking.

        Parameters
        ----------
        beam : Beam object

        """
        if beam.mpi_switch:
            self.bunch_index = beam.mpi.bunch_num # Number of the tracked bunch in this processor
            
        self.distance = beam.distance_between_bunches
        self.valid_bunch_index = beam.bunch_index
        self.tracking = True
        self.nturn = 0
    
    def track(self, beam):
        """
        Track a Beam object through the CavityResonator object.
        
        Can be used with or without mpi.
        If used with mpi, beam.mpi.share_distributions must be called before.
        
        The beam phasor is given at t=0 (synchronous particle) of the first 
        non empty bunch.

        Parameters
        ----------
        beam : Beam object

        """
        
        if self.tracking is False:
            self.init_tracking(beam)
        
        for index, bunch in enumerate(beam):
            
            if beam.filling_pattern[index]:
                
                if beam.mpi_switch:
                    # get rank of bunch n° index
                    rank = beam.mpi.bunch_to_rank(index)
                    # mpi -> get shared bunch profile for current bunch
                    center = beam.mpi.tau_center[rank]
                    profile = beam.mpi.tau_profile[rank]
                    bin_length = center[1]-center[0]
                    charge_per_mp = beam.mpi.charge_per_mp_all[rank]
                    if index == self.bunch_index:
                        sorted_index = beam.mpi.tau_sorted_index
                else:
                    # no mpi -> get bunch profile for current bunch
                    if len(bunch) != 0:
                        (bins, sorted_index, profile, center) = bunch.binning(n_bin=self.n_bin)
                        bin_length = center[1]-center[0]
                        charge_per_mp = bunch.charge_per_mp
                        self.bunch_index = index
                    else:
                        # Update filling pattern
                        beam.update_filling_pattern()
                        beam.update_distance_between_bunches()
                        # save beam phasor value
                        self.beam_phasor_record[index] = self.beam_phasor
                        # phasor decay to be at t=0 of the next bunch
                        self.phasor_decay(self.ring.T1, ref_frame="beam")
                        continue
                
                energy_change = bunch["tau"]*0
                
                # remove part of beam phasor decay to be at the start of the binning (=bins[0])
                self.phasor_decay(center[0] - bin_length/2, ref_frame="beam")
                
                if index != self.bunch_index:
                    self.phasor_evol(profile, bin_length, charge_per_mp, ref_frame="beam")
                else:
                    # modify beam phasor
                    for i, center0 in enumerate(center):
                        mp_per_bin = profile[i]
                        
                        if mp_per_bin == 0:
                            self.phasor_decay(bin_length, ref_frame="beam")
                            continue
                        
                        ind = (sorted_index == i)
                        phase = self.m * self.ring.omega1 * (center0 + self.ring.T1* (index + self.ring.h * self.nturn))
                        Vgene = self.Vg*np.cos(phase + self.theta_g)
                        Vbeam = np.real(self.beam_phasor)
                        Vtot = Vgene + Vbeam - charge_per_mp*self.loss_factor*mp_per_bin
                        energy_change[ind] = Vtot / self.ring.E0
    
                        self.beam_phasor -= 2*charge_per_mp*self.loss_factor*mp_per_bin
                        self.phasor_decay(bin_length, ref_frame="beam")
                
                # phasor decay to be at t=0 of the current bunch (=-1*bins[-1])
                self.phasor_decay(-1 * (center[-1] + bin_length/2), ref_frame="beam")
                
                if index == self.bunch_index:
                    # apply kick
                    bunch["delta"] += energy_change
            
            # save beam phasor value
            self.beam_phasor_record[index] = self.beam_phasor
            
            # phasor decay to be at t=0 of the next bunch
            self.phasor_decay(self.ring.T1, ref_frame="beam")
                
        self.nturn += 1
                
        
    def init_phasor_track(self, beam):
        """
        Initialize the beam phasor for a given beam distribution using a
        tracking like method.
        
        Follow the same steps as the track method but in the "rf" reference 
        frame and without any modifications on the beam.

        Parameters
        ----------
        beam : Beam object

        """        
        if self.tracking is False:
            self.init_tracking(beam)
            
        n_turn = int(self.filling_time/self.ring.T0*10)
        
        for i in range(n_turn):
            for j, bunch in enumerate(beam.not_empty):
                
                index = self.valid_bunch_index[j]
                
                if beam.mpi_switch:
                    # get shared bunch profile for current bunch
                    center = beam.mpi.tau_center[j]
                    profile = beam.mpi.tau_profile[j]
                    bin_length = center[1]-center[0]
                    charge_per_mp = beam.mpi.charge_per_mp_all[j]
                else:
                    if i == 0:
                        # get bunch profile for current bunch
                        (bins, sorted_index, profile, center) = bunch.binning(n_bin=self.n_bin)
                        if j == 0:
                            self.profile_save = np.zeros((len(beam),len(profile),))
                            self.center_save = np.zeros((len(beam),len(center),))
                        self.profile_save[j,:] = profile
                        self.center_save[j,:] = center
                    else:
                        profile = self.profile_save[j,:]
                        center = self.center_save[j,:]
                        
                    bin_length = center[1]-center[0]
                    charge_per_mp = bunch.charge_per_mp
                
                self.phasor_decay(center[0] - bin_length/2, ref_frame="rf")
                self.phasor_evol(profile, bin_length, charge_per_mp, ref_frame="rf")
                self.phasor_decay(-1 * (center[-1] + bin_length/2), ref_frame="rf")
                self.phasor_decay( (self.distance[index] * self.ring.T1), ref_frame="rf")
            
    def phasor_decay(self, time, ref_frame="beam"):
        """
        Compute the beam phasor decay during a given time span, assuming that 
        no particles are crossing the cavity during the time span.

        Parameters
        ----------
        time : float
            Time span in [s], can be positive or negative.
        ref_frame : string, optional
            Reference frame to be used, can be "beam" or "rf".

        """
        if ref_frame == "beam":
            delta = self.wr
        elif ref_frame == "rf":
            delta = (self.wr - self.m*self.ring.omega1)
        self.beam_phasor = self.beam_phasor * np.exp((-1/self.filling_time +
                                  1j*delta)*time)
        
    def phasor_evol(self, profile, bin_length, charge_per_mp, ref_frame="beam"):
        """
        Compute the beam phasor evolution during the crossing of a bunch using 
        an analytic formula [1].
        
        Assume that the phasor decay happens before the beam loading.

        Parameters
        ----------
        profile : array
            Longitudinal profile of the bunch in [number of macro-particle].
        bin_length : float
            Length of a bin in [s].
        charge_per_mp : float
            Charge per macro-particle in [C].
        ref_frame : string, optional
            Reference frame to be used, can be "beam" or "rf".
            
        References
        ----------
        [1] mbtrack2 manual.
            
        """
        if ref_frame == "beam":
            delta = self.wr
        elif ref_frame == "rf":
            delta = (self.wr - self.m*self.ring.omega1)
            
        n_bin = len(profile)
        
        # Phasor decay during crossing time
        deltaT = n_bin*bin_length
        self.phasor_decay(deltaT, ref_frame)
        
        # Phasor evolution due to induced voltage by marco-particles
        k = np.arange(0, n_bin)
        var = np.exp( (-1/self.filling_time + 1j*delta) * 
                      (n_bin-k) * bin_length )
        sum_tot = np.sum(profile * var)
        sum_val = -2 * sum_tot * charge_per_mp * self.loss_factor
        self.beam_phasor += sum_val
        
    def init_phasor(self, beam):
        """
        Initialize the beam phasor for a given beam distribution using an
        analytic formula [1].
        
        No modifications on the Beam object.

        Parameters
        ----------
        beam : Beam object
            
        References
        ----------
        [1] mbtrack2 manual.

        """
        
        # Initialization
        if self.tracking is False:
            self.init_tracking(beam)
        
        N = self.n_bin - 1
        delta = (self.wr - self.m*self.ring.omega1)
        n_turn = int(self.filling_time/self.ring.T0*10)
        
        T = np.ones(self.ring.h)*self.ring.T1
        bin_length = np.zeros(self.ring.h)
        charge_per_mp = np.zeros(self.ring.h)
        profile = np.zeros((N, self.ring.h))
        center = np.zeros((N, self.ring.h))
        
        # Gather beam distribution data
        for j, bunch in enumerate(beam.not_empty):
            index = self.valid_bunch_index[j]
            if beam.mpi_switch:
                beam.mpi.share_distributions(beam, n_bin=self.n_bin)
                center[:,index] = beam.mpi.tau_center[j]
                profile[:,index] = beam.mpi.tau_profile[j]
                bin_length[index] = center[1, index]-center[0, index]
                charge_per_mp[index] = beam.mpi.charge_per_mp_all[j]
            else:
                (bins, sorted_index, profile[:, index], center[:, index]) = bunch.binning(n_bin=self.n_bin)
                bin_length[index] = center[1, index]-center[0, index]
                charge_per_mp[index] = bunch.charge_per_mp
            T[index] -= (center[-1, index] + bin_length[index]/2)
            if index != 0:
                T[index - 1] += (center[0, index] - bin_length[index]/2)
        T[self.ring.h - 1] += (center[0, 0] - bin_length[0]/2)

        # Compute matrix coefficients
        k = np.arange(0, N)
        Tkj = np.zeros((N, self.ring.h))
        for j in range(self.ring.h):
            sum_t = np.array([T[n] + N*bin_length[n] for n in range(j+1,self.ring.h)])
            Tkj[:,j] = (N-k)*bin_length[j] + T[j] + np.sum(sum_t)
            
        var = np.exp( (-1/self.filling_time + 1j*delta) * Tkj )
        sum_tot = np.sum((profile*charge_per_mp) * var)
        
        # Use the formula n_turn times
        for i in range(n_turn):
            # Phasor decay during one turn
            self.phasor_decay(self.ring.T0, ref_frame="rf")
            # Phasor evolution due to induced voltage by marco-particles during one turn
            sum_val = -2 * sum_tot * self.loss_factor
            self.beam_phasor += sum_val
        
        # Replace phasor at t=0 (synchronous particle) of the first non empty bunch.
        idx0 = self.valid_bunch_index[0]
        self.phasor_decay(center[-1,idx0] + bin_length[idx0]/2, ref_frame="rf")
    
    @property
    def generator_phasor(self):
        """Generator phasor in [V]"""
        return self.Vg*np.exp(1j*self.theta_g)
    
    @property
    def cavity_phasor(self):
        """Cavity total phasor in [V]"""
        return self.generator_phasor + self.beam_phasor
    
    @property
    def cavity_phasor_record(self):
        """Last cavity phasor value of each bunch in [V]"""
        return self.generator_phasor + self.beam_phasor_record
    
    @property
    def cavity_voltage(self):
        """Cavity total voltage in [V]"""
        return np.abs(self.cavity_phasor)
    
    @property
    def cavity_phase(self):
        """Cavity total phase in [rad]"""
        return np.angle(self.cavity_phasor)
    
    @property
    def beam_voltage(self):
        """Beam loading voltage in [V]"""
        return np.abs(self.beam_phasor)
    
    @property
    def beam_phase(self):
        """Beam loading phase in [rad]"""
        return np.angle(self.beam_phasor)
    
    @property
    def loss_factor(self):
        """Cavity loss factor in [V/C]"""
        return self.wr*self.Rs/(2 * self.Q)

    @property
    def m(self):
        """Harmonic number of the cavity"""
        return self._m

    @m.setter
    def m(self, value):
        self._m = value
        
    @property
    def Ncav(self):
        """Number of cavities"""
        return self._Ncav

    @Ncav.setter
    def Ncav(self, value):
        self._Ncav = value
        
    @property
    def Rs_per_cavity(self):
        """Shunt impedance of a single cavity in [Ohm], defined as 
        0.5*Vc*Vc/Pc."""
        return self._Rs_per_cavity

    @Rs_per_cavity.setter
    def Rs_per_cavity(self, value):
        self._Rs_per_cavity = value

    @property
    def Rs(self):
        """Shunt impedance [ohm]"""
        return self.Rs_per_cavity * self.Ncav

    @Rs.setter
    def Rs(self, value):
        self.Rs_per_cavity = value / self.Ncav

    @property
    def Q(self):
        """Quality factor"""
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value

    @property
    def QL(self):
        """Loaded quality factor"""
        return self._QL

    @QL.setter
    def QL(self, value):
        self._QL = value
        self._beta = self.Q/self.QL - 1

    @property
    def beta(self):
        """Coupling coefficient"""
        return self._beta

    @beta.setter
    def beta(self, value):
        self.QL = self.Q/(1 + value)

    @property
    def detune(self):
        """Cavity detuning [Hz] - defined as (fr - m*f1)"""
        return self._detune

    @detune.setter
    def detune(self, value):
        self._detune = value
        self._fr = self.detune + self.m*self.ring.f1
        self._wr = self.fr*2*np.pi
        self._psi = np.arctan(self.QL*(self.fr/(self.m*self.ring.f1) -
                                       (self.m*self.ring.f1)/self.fr))

    @property
    def fr(self):
        """Resonance frequency of the cavity in [Hz]"""
        return self._fr

    @fr.setter
    def fr(self, value):
        self.detune = value - self.m*self.ring.f1

    @property
    def wr(self):
        """Angular resonance frequency in [Hz.rad]"""
        return self._wr

    @wr.setter
    def wr(self, value):
        self.detune = (value - self.m*self.ring.f1)*2*np.pi

    @property
    def psi(self):
        """Tuning angle in [rad]"""
        return self._psi

    @psi.setter
    def psi(self, value):
        delta = (self.ring.f1*self.m*np.tan(value)/self.QL)**2 + 4*(self.ring.f1*self.m)**2
        fr = (self.ring.f1*self.m*np.tan(value)/self.QL + np.sqrt(delta))/2
        self.detune = fr - self.m*self.ring.f1
        
    @property
    def filling_time(self):
        """Cavity filling time in [s]"""
        return 2*self.QL/self.wr
    
    @property
    def Pc(self):
        """Power dissipated in the cavity walls in [W]"""
        return self.Vc**2 / (2 * self.Rs)
    
    def Pb(self, I0):
        """
        Return power transmitted to the beam in [W] - near Eq. (4.2.3) in [1].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Power transmitted to the beam in [W].

        """
        return I0 * self.Vc * np.cos(self.theta)
    
    def Pr(self, I0):
        """
        Power reflected back to the generator in [W].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Power reflected back to the generator in [W].

        """
        return self.Pg - self.Pb(I0) - self.Pc

    def Vbr(self, I0):
        """
        Return beam voltage at resonance in [V].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Beam voltage at resonance in [V].

        """
        return 2*I0*self.Rs/(1+self.beta)
    
    def Vb(self, I0):
        """
        Return beam voltage in [V].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        float
            Beam voltage in [V].

        """
        return self.Vbr(I0)*np.cos(self.psi)
    
    def Z(self, f):
        """Cavity impedance in [Ohm] for a given frequency f in [Hz]"""
        return self.Rs/(1 + 1j*self.QL*(self.fr/f - f/self.fr))
    
    def set_optimal_detune(self, I0):
        """
        Set detuning to optimal conditions - second Eq. (4.2.1) in [1].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        """
        self.psi = np.arctan(-self.Vbr(I0)/self.Vc*np.sin(self.theta))
        
    def set_optimal_coupling(self, I0):
        """
        Set coupling to optimal value - Eq. (4.2.3) in [1].

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        """
        self.beta = 1 + (2 * I0 * self.Rs * np.cos(self.theta) / 
                         self.Vc)
                
    def set_generator(self, I0):
        """
        Set generator parameters (Pg, Vgr, theta_gr, Vg and theta_g) for a 
        given current and set of parameters.

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        """
        
        # Generator power [W] - Eq. (4.1.2) [1] corrected with factor (1+beta)**2 instead of (1+beta**2)
        self.Pg = self.Vc**2*(1+self.beta)**2/(2*self.Rs*4*self.beta*np.cos(self.psi)**2)*(
            (np.cos(self.theta) + 2*I0*self.Rs/(self.Vc*(1+self.beta))*np.cos(self.psi)**2 )**2 + 
            (np.sin(self.theta) + 2*I0*self.Rs/(self.Vc*(1+self.beta))*np.cos(self.psi)*np.sin(self.psi) )**2)
        # Generator voltage at resonance [V] - Eq. (3.2.2) [1]
        self.Vgr = 2*self.beta**(1/2)/(1+self.beta)*(2*self.Rs*self.Pg)**(1/2)
        # Generator phase at resonance [rad] - from Eq. (4.1.1)
        self.theta_gr = np.arctan((self.Vc*np.sin(self.theta) + self.Vbr(I0)*np.cos(self.psi)*np.sin(self.psi))/
                    (self.Vc*np.cos(self.theta) + self.Vbr(I0)*np.cos(self.psi)**2)) - self.psi
        # Generator voltage [V]
        self.Vg = self.Vgr*np.cos(self.psi)
        # Generator phase [rad]
        self.theta_g = self.theta_gr + self.psi
        
    def plot_phasor(self, I0):
        """
        Plot phasor diagram showing the vector addition of generator and beam 
        loading voltage.

        Parameters
        ----------
        I0 : float
            Beam current in [A].
            
        Returns
        -------
        Figure.

        """

        def make_legend_arrow(legend, orig_handle,
                              xdescent, ydescent,
                              width, height, fontsize):
            p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
            return p

        fig = plt.figure()
        ax= fig.add_subplot(111, polar=True)
        ax.set_rmax(max([1.2,self.Vb(I0)/self.Vc*1.2,self.Vg/self.Vc*1.2]))
        arr1 = ax.arrow(self.theta, 0, 0, 1, alpha = 0.5, width = 0.015,
                         edgecolor = 'black', lw = 2)

        arr2 = ax.arrow(self.psi + np.pi, 0, 0,self.Vb(I0)/self.Vc, alpha = 0.5, width = 0.015,
                         edgecolor = 'red', lw = 2)

        arr3 = ax.arrow(self.theta_g, 0, 0,self.Vg/self.Vc, alpha = 0.5, width = 0.015,
                         edgecolor = 'blue', lw = 2)

        ax.set_rticks([])  # less radial ticks
        plt.legend([arr1,arr2,arr3], ['Vc','Vb','Vg'],handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),})
        
        return fig
        
    def is_DC_Robinson_stable(self, I0):
        """
        Check DC Robinson stability - Eq. (6.1.1) [1]

        Parameters
        ----------
        I0 : float
            Beam current in [A].

        Returns
        -------
        bool

        """
        return 2*self.Vc*np.sin(self.theta) + self.Vbr(I0)*np.sin(2*self.psi) > 0
    
    def plot_DC_Robinson_stability(self, detune_range = [-1e5,1e5]):
        """
        Plot DC Robinson stability limit.

        Parameters
        ----------
        detune_range : list or array, optional
            Range of tuning to plot in [Hz].

        Returns
        -------
        Figure.

        """
        old_detune = self.psi
        
        x = np.linspace(detune_range[0],detune_range[1],1000)
        y = []
        for i in range(0,x.size):
            self.detune = x[i]
            y.append(-self.Vc*(1+self.beta)/(self.Rs*np.sin(2*self.psi))*np.sin(self.theta)) # droite de stabilité
            
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(x,y)
        ax.set_xlabel("Detune [Hz]")
        ax.set_ylabel("Threshold current [A]")
        ax.set_title("DC Robinson stability limit")
        
        self.psi = old_detune
        
        return fig
        
    def VRF(self, z, I0, F = 1, PHI = 0):
        """Total RF voltage taking into account form factor amplitude F and form factor phase PHI"""
        return self.Vg*np.cos(self.ring.k1*self.m*z + self.theta_g) - self.Vb(I0)*F*np.cos(self.ring.k1*self.m*z + self.psi - PHI)
    
    def dVRF(self, z, I0, F = 1, PHI = 0):
        """Return derivative of total RF voltage taking into account form factor amplitude F and form factor phase PHI"""
        return -1*self.Vg*self.ring.k1*self.m*np.sin(self.ring.k1*self.m*z + self.theta_g) + self.Vb(I0)*F*self.ring.k1*self.m*np.sin(self.ring.k1*self.m*z + self.psi - PHI)
    
    def ddVRF(self, z, I0, F = 1, PHI = 0):
        """Return the second derivative of total RF voltage taking into account form factor amplitude F and form factor phase PHI"""
        return -1*self.Vg*(self.ring.k1*self.m)**2*np.cos(self.ring.k1*self.m*z + self.theta_g) + self.Vb(I0)*F*(self.ring.k1*self.m)**2*np.cos(self.ring.k1*self.m*z + self.psi - PHI)
        
    def deltaVRF(self, z, I0, F = 1, PHI = 0):
        """Return the generator voltage minus beam loading voltage taking into account form factor amplitude F and form factor phase PHI"""
        return -1*self.Vg*(self.ring.k1*self.m)**2*np.cos(self.ring.k1*self.m*z + self.theta_g) - self.Vb(I0)*F*(self.ring.k1*self.m)**2*np.cos(self.ring.k1*self.m*z + self.psi - PHI)