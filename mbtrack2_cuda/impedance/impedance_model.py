# -*- coding: utf-8 -*-
"""
Module where the ImpedanceModel class is defined.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mbtrack2_cuda.utilities.misc import (beam_loss_factor, effective_impedance, 
                                     double_sided_impedance)
from mbtrack2_cuda.utilities.spectrum import (beam_spectrum, 
                                         gaussian_bunch_spectrum, 
                                         spectral_density)
from mbtrack2_cuda.impedance.wakefield import WakeField
from mbtrack2_cuda.tracking.element import Element

class ImpedanceModel(Element):
    """
    Define the impedance model of the machine.
    
    Parameters
    ----------
    ring : Synchrotron object
    wakefield_list : list of WakeField objects
        WakeFields to add to the model.
    wakefiled_positions : list
        Longitudinal positions corresponding to the added Wakfields.
    
    Attributes
    ----------
    wakefields : list of WakeField objects
        WakeFields in the model.
    positions : array
        WakeFields positions.
    names : array
        Names of (unique) WakeField objects.
    sum : WakeField
        Sum of every WakeField in the model.
    sum_"name" : WakeField
        Sum of every Wakefield with the same "name".
    sum_names : array
        Names of attributes where the WakeFields are summed by name.    
    
    Methods
    -------
    add(wakefield_list, wakefiled_positions)
        Add elements to the model.
    add_multiple_elements(wakefield, wakefiled_positions)
        Add the same element at different locations to the model.
    sum_elements()
        Sum all WakeFields into self.sum.
    update_name_list()
        Update self.names with uniques names of self.wakefields.
    find_wakefield(name)
        Return indexes of WakeFields with the same name in self.wakefields.
    sum_by_name(name)
        Sum the elements with the same name in the model into sum_name.
    sum_by_name_all(name)
        Sum all the elements with the same name in the model into sum_name.
    plot_area(Z_type="Zlong", component="real", sigma=None, attr_list=None)
        Plot the contributions of different kind of WakeFields.
    save(file)
        Save impedance model to file.
    load(file)
        Load impedance model from file.
    """
    
    def __init__(self, ring, wakefield_list=None, wakefiled_positions=None):
        self.ring = ring
        self.optics = self.ring.optics
        self.wakefields = []
        self.positions = np.array([])
        self.names = np.array([])
        self.sum_names = np.array([])
        self.add(wakefield_list, wakefiled_positions)
        
    def track(self, beam):
        """
        Track a beam object through this Element.
        
        Parameters
        ----------
        beam : Beam object
        """
        raise NotImplementedError
        
    def sum_elements(self):
        """Sum all WakeFields into self.sum"""
        beta = self.optics.beta(self.positions)
        self.sum = WakeField.add_several_wakefields(self.wakefields, beta)
        if "sum" not in self.sum_names:
            self.sum_names = np.append(self.sum_names, "sum")
    
    def update_name_list(self):
        """Update self.names with uniques names of self.wakefields."""
        for wakefield in self.wakefields:
            if wakefield.name is None:
                continue
            if wakefield.name not in self.names:
                self.names = np.append(self.names, wakefield.name)
                
    def count_elements(self):
        """Count number of each type of WakeField in the model."""
        self.count = np.zeros(len(self.names))
        for wakefield in self.wakefields:
            if wakefield.name is not None:
                self.count += (wakefield.name == self.names).astype(int)
                
    def find_wakefield(self, name):
        """
        Return indexes of WakeFields with the same name in self.wakefields.

        Parameters
        ----------
        name : str
            WakeField name.

        Returns
        -------
        index : list
            Index of positions in self.wakefields.

        """
        index = []
        for i, wakefield in enumerate(self.wakefields):
            if wakefield.name == name:
                index.append(i)
        return index

    def sum_by_name(self, name):
        """
        Sum the elements with the same name in the model into sum_name.
        
        Parameters
        ----------
        name : str
            Name of the WakeField to sum.
        """
        attribute_name = "sum_" + name
        index = self.find_wakefield(name)
        beta = self.optics.beta(self.positions[index])
        wakes = []
        for i in index:
            wakes.append(self.wakefields[i])
        wake_sum = WakeField.add_several_wakefields(wakes, beta)
        setattr(self, attribute_name, wake_sum)
        if attribute_name not in self.sum_names:
            self.sum_names = np.append(self.sum_names, attribute_name)
            
    def sum_by_name_all(self):
        """
        Sum all the elements with the same name in the model into sum_name.
        """
        for name in self.names:
            self.sum_by_name(name)
                    
    def add(self, wakefield_list, wakefiled_positions):
        """
        Add elements to the model.

        Parameters
        ----------
        wakefield_list : list of WakeField objects
            WakeFields to add to the model.
        wakefiled_positions : list
            Longitudinal positions corresponding to the added Wakfields.
        """
        if (wakefield_list is not None) and (wakefiled_positions is not None):
            for wakefield in wakefield_list:
                self.wakefields.append(wakefield)
                
            for position in wakefiled_positions:
                self.positions = np.append(self.positions, position)
                
        self.update_name_list()
                
    def add_multiple_elements(self, wakefield, wakefiled_positions):
        """
        Add the same element at different locations to the model.

        Parameters
        ----------
        WakeField : WakeField object
            WakeField to add to the model.
        wakefiled_positions : list
            Longitudinal positions corresponding to the added Wakfield.
        """
        for position in wakefiled_positions:
            self.positions = np.append(self.positions, position)
            self.wakefields.append(wakefield)
            
        self.update_name_list()
            
    def plot_area(self, Z_type="Zlong", component="real", sigma=None, 
                  attr_list=None, zoom=False):
        """
        Plot the contributions of different kind of WakeFields.

        Parameters
        ----------
        Z_type : str, optional
            Type of impedance to plot.
        component : str, optional
            Component to plot, can be "real" or "imag". 
        sigma : float, optional
            RMS bunch length in [s] to use for the spectral density. If equal
            to None, the spectral density is not plotted.
        attr_list : list or array of str, optional
            Attributes to plot.
        zoom : bool
            If True, add a zoomed plot on top right corner.

        """
        if attr_list is None:
            attr_list = self.sum_names[self.sum_names != "sum"]
        
        # manage legend
        Ztype_dict = {"Zlong":0, "Zxdip":1, "Zydip":2, "Zxquad":3, "Zyquad":4}
        scale = [1e-3, 1e-6, 1e-6, 1e-6, 1e-6]
        label_list =  [r"$Z_{long} \; [k\Omega]$",
                       r"$\sum_{j} \beta_{x,j} Z_{x,j}^{Dip} \; [M\Omega]$",
                       r"$\sum_{j} \beta_{y,j} Z_{y,j}^{Dip} \; [M\Omega]$",
                       r"$\sum_{j} \beta_{x,j} Z_{x,j}^{Quad} \; [M\Omega]$",
                       r"$\sum_{j} \beta_{y,j} Z_{y,j}^{Quad} \; [M\Omega]$"]
        leg = Ztype_dict[Z_type]
        
        # sort plot by decresing area size        
        area = np.zeros((len(attr_list),))
        for index, attr in enumerate(attr_list):
            try:
                sum_imp = getattr(getattr(self, attr), Z_type)
                area[index] = trapz(sum_imp.data[component], sum_imp.data.index)
            except AttributeError:
                pass
        sorted_index = area.argsort()[::-1]
        
        # Init fig
        fig = plt.figure()
        ax = fig.gca()
        zero_impedance = getattr(self.sum, Z_type)*0
        total_imp = 0
        legend = []
        
        if sigma is not None:
            legend.append("Spectral density for sigma = " + str(sigma) + " s")
        
        # Main plot
        for index in  sorted_index:
            attr = attr_list[index]
            # Set all impedances with common indexes using + zero_impedance
            try:
                sum_imp = getattr(getattr(self, attr), Z_type) + zero_impedance
                ax.fill_between(sum_imp.data.index*1e-9, total_imp, 
                                total_imp + sum_imp.data[component]*scale[leg])
                total_imp += sum_imp.data[component]*scale[leg]
                if attr[:4] == "sum_":
                    legend.append(attr[4:])
                else:
                    legend.append(attr)
            except AttributeError:
                pass
            
        if sigma is not None:
            spect = spectral_density(zero_impedance.data.index, sigma)
            spect = spect/spect.max()*total_imp.max()
            ax.plot(sum_imp.data.index*1e-9, spect, 'r', linewidth=2.5)
        
        ax.legend(legend, loc="upper left")            
        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel(label_list[leg] + " - " + component + " part")
        ax.set_title(label_list[leg] + " - " + component + " part")
        
        if zoom is True:
            in_ax = inset_axes(ax,
                            width="30%", # width = 30% of parent_bbox
                            height=1.5, # height : 1 inch
                            loc=1)
            
            total_imp = 0
            for index in  sorted_index:
                attr = attr_list[index]
                # Set all impedances with common indexes using + zero_impedance
                try:
                    sum_imp = getattr(getattr(self, attr), Z_type) + zero_impedance
                    in_ax.fill_between(sum_imp.data.index*1e-3, total_imp, 
                                    total_imp + sum_imp.data[component]*1e-9)
                    total_imp += sum_imp.data[component]*1e-9
                except AttributeError:
                    pass
            in_ax.set_xlim([0, 200])
            in_ax.set_xlabel("Frequency [kHz]")
            in_ax.set_ylabel(r"$[G\Omega]$")
                
        return fig
            
    def effective_impedance(self, m, mu, sigma, M, tuneS, xi=None, 
                            mode="Hermite"):
        """
        Compute the longitudinal and transverse effective impedance.

        Parameters
        ----------
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
        summary : DataFrame
            Longitudinal and transverse effective impedance.

        """

        attr_list = self.sum_names
        
        eff_array = np.zeros((len(attr_list),3), dtype=complex)
        
        for i, attr in enumerate(attr_list):
            try:
                impedance = getattr(getattr(self, attr), "Zlong")
                eff_array[i,0] = effective_impedance(self.ring, impedance, 
                                                     m, mu, sigma, M, tuneS,
                                                     xi, mode)
            except AttributeError:
                pass
            
            try:
                impedance = getattr(getattr(self, attr), "Zxdip")
                eff_array[i,1] = effective_impedance(self.ring, impedance, 
                                                     m, mu, sigma, M, tuneS,
                                                     xi, mode)
            except AttributeError:
                pass
            
            try:
                impedance = getattr(getattr(self, attr), "Zydip")
                eff_array[i,2] = effective_impedance(self.ring, impedance, 
                                                     m, mu, sigma, M, tuneS,
                                                     xi, mode)
            except AttributeError:
                pass
            
        eff_array[:,0] = eff_array[:,0]*self.ring.omega0*1e3
        eff_array[:,1] = eff_array[:,1]*1e-3
        eff_array[:,2] = eff_array[:,2]*1e-3
        
        summary = pd.DataFrame(eff_array, index=attr_list, 
                               columns=["Z/n [mOhm]", 
                                        "sum betax x Zxeff [kOhm]", 
                                        "sum betay x Zyeff [kOhm]"])
        
        return summary


    def energy_loss(self, sigma, M, bunch_spacing, I, n_points=10e6):
        """
        Compute the beam and bunch loss factor and energy losses for each type 
        of element in the model assuming Gaussian bunches and constant spacing
        between bunches.

        Parameters
        ----------
        sigma : float
            RMS bunch length in [s].
        M : int
            Number of bunches in the beam.
        bunch_spacing : float
            Time between two bunches in [s].
        I : float
            Total beam current in [A].
        n_points : float, optional
            Number of points used in the frequency spectrums.

        Returns
        -------
        summary : Dataframe
            Contains the beam and bunch loss factor and energy loss for the 
            full model and for each type of different component.

        """
        
        fmax = self.sum.Zlong.data.index.max()
        fmin = self.sum.Zlong.data.index.min()
        
        Q = I*self.ring.T0/M
        
        if fmin >= 0:
            fmin = -1*fmax
        f = np.linspace(fmin, fmax, int(n_points))

        beam_spect = beam_spectrum(f, M, bunch_spacing, sigma= sigma)
        
        bunch_spect = gaussian_bunch_spectrum(f, sigma)
        
        attr_list = self.sum_names
        
        loss_array = np.zeros((len(attr_list),2))
        
        for i, attr in enumerate(attr_list):
            try:
                impedance = getattr(getattr(self, attr), "Zlong")
                loss_array[i,0] = beam_loss_factor(impedance, f, beam_spect, self.ring)
                loss_array[i,1] = beam_loss_factor(impedance, f, bunch_spect, self.ring)
            except AttributeError:
                pass

        loss_array = loss_array*1e-12
        summary = pd.DataFrame(loss_array, index=attr_list, 
                               columns=["loss factor (beam) [V/pC]", "loss factor (bunch) [V/pC]"])
        
        summary["P (beam) [W]"] = summary["loss factor (beam) [V/pC]"]*1e12*Q**2/(self.ring.T0)
        summary["P (bunch) [W]"] = summary["loss factor (bunch) [V/pC]"]*1e12*Q**2/(self.ring.T0)*M
                
        return summary
    
    def power_loss_spectrum(self, sigma, M, bunch_spacing, I, n_points=10e6, 
                            max_overlap=False,plot=False):
        """
        Compute the power loss spectrum of the summed longitudinal impedance 
        as in Eq. (4) of [1].
        
        Assume Gaussian bunches and constant spacing between bunches.

        Parameters
        ----------
        sigma : float
            RMS bunch length in [s].
        M : int
            Number of bunches in the beam.
        bunch_spacing : float
            Time between two bunches in [s].
        I : float
            Total beam current in [A].
        n_points : float, optional
            Number of points used in the frequency spectrum.
        max_overlap : bool, optional
            If True, the bunch spectrum (scaled to the number of bunches) is 
            used instead of the beam spectrum to compute the maximum value of 
            the power loss spectrum at each frequency. Should only be used to 
            maximise the power loss at a given frequency (e.g. for HOMs) and 
            not for the full spectrum.
        plot : bool, optional
            If True, plots:
                - the overlap between the real part of the longitudinal impedance 
                and the beam spectrum.
                - the power loss spectrum.

        Returns
        -------
        pf0 : array
            Frequency points.
        power_loss : array
            Power loss spectrum in [W].
            
        References
        ----------
        [1] : L. Teofili, et al. "A Multi-Physics Approach to Simulate the RF 
        Heating 3D Power Map Induced by the Proton Beam in a Beam Intercepting 
        Device", in IPAC'18, 2018, doi:10.18429/JACoW-IPAC2018-THPAK093

        """
        
        impedance = self.sum.Zlong
        fmax = impedance.data.index.max()
        fmin = impedance.data.index.min()
        
        Q = I*self.ring.T0/M
            
        if fmin >= 0:
            fmin = -1*fmax
            
        frequency = np.linspace(fmin, fmax, int(n_points))
        if max_overlap is False:
            spectrum =  beam_spectrum(frequency, M, bunch_spacing, sigma)
        else:
            spectrum = gaussian_bunch_spectrum(frequency, sigma)*M
        
        pmax = np.floor(fmax/self.ring.f0)
        pmin = np.floor(fmin/self.ring.f0)
        
        if pmin >= 0:
            double_sided_impedance(impedance)
            pmin = -1*pmax
        
        p = np.arange(pmin+1,pmax)    
        pf0 = p*self.ring.f0
        ReZ = np.real(impedance(pf0))
        spectral_density = np.abs(spectrum)**2
        # interpolation of the spectrum is needed to avoid problems liked to 
        # division by 0
        # computing the spectrum directly to the frequency points gives
        # wrong results
        spect = interp1d(frequency, spectral_density)
        power_loss = (self.ring.f0 * Q)**2 * ReZ * spect(pf0)
        
        if plot is True:
            fig, ax = plt.subplots()
            twin = ax.twinx()
            p1, = ax.plot(pf0, ReZ, color="r",label="Re[Z] [Ohm]")
            p2, = twin.plot(pf0, spect(pf0)*(self.ring.f0 * Q)**2, color="b", 
                            label="Beam spectrum [a.u.]")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Re[Z] [Ohm]")
            twin.set_ylabel("Beam spectrum [a.u.]")
            plots = [p1, p2]
            ax.legend(handles=plots, loc="best")
            
            fig, ax = plt.subplots()
            ax.plot(pf0, power_loss)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Power loss [W]")
        
        return pf0, power_loss
    
    def save(self, file):
        """
        Save impedance model to file.
        
        The same pandas version is needed on both saving and loading computer
        for the pickle to work.

        Parameters
        ----------
        file : str
            File where the impedance model is saved.

        Returns
        -------
        None.

        """
        to_save = {"wakefields":self.wakefields,
                   "positions":self.positions}
        with open(file,"wb") as f:
            pickle.dump(to_save, f)
    
    def load(self, file):
        """
        Load impedance model from file.
        
        The same pandas version is needed on both saving and loading computer
        for the pickle to work.

        Parameters
        ----------
        file : str
            File where the impedance model is saved.

        Returns
        -------
        None.

        """
        
        with open(file, 'rb') as f:
            to_load = pickle.load(f)
            
        self.wakefields = to_load["wakefields"]
        self.positions = to_load["positions"]  
        self.sum_elements()
        self.sum_by_name_all()
