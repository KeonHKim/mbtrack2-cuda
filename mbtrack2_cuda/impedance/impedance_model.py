# -*- coding: utf-8 -*-
"""
Module where the ImpedanceModel class is defined.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import pickle
from copy import deepcopy
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mbtrack2_cuda.utilities.misc import (beam_loss_factor, effective_impedance,
                                     double_sided_impedance)
from mbtrack2_cuda.utilities.spectrum import (beam_spectrum,
                                         gaussian_bunch_spectrum,
                                         spectral_density)
from mbtrack2_cuda.impedance.wakefield import WakeField, WakeFunction


class ImpedanceModel():
    """
    Define the impedance model of the machine.

    The model must be completed with successive add(...) and add_global(...)
    calls, then compute_sum() must be run.

    The transverse impedance and wake functions are beta weighted and divided
    by the beta at the tracking location (ring.optics.local_beta).

    Parameters
    ----------
    ring : Synchrotron object

    Attributes
    ----------
    wakefields : list of WakeField objects
        WakeFields in the model.
    positions : list of arrays
        Positions corresponding the different WakeFields.
    names : list of str
        Names of the WakeField objects.
    sum : WakeField
        Sum of every WakeField in the model weigthed by beta functions.
    sum_"name" : WakeField
        Sum of the "name" Wakefield weigthed by beta functions.
    sum_names : array
        Names of attributes where the WakeFields are summed by name.
    globals : list of WakeField objects
        Globals WakeFields in the model.
    globals_names : list of str
        Names of the global WakeFields objects.

    Methods
    -------
    add(wakefield, positions, name)
        Add the same WakeField object at different locations to the model.
    add_global(wakefield, name)
        Add a "global" WakeField object which will added to the sum WakeField
        but not weighted by beta functions.
    sum_beta(wake, beta)
        Weight a WakeField object by an array of beta functions.
    compute_sum_names()
        Compute the weighted WakeField for each WakeField object type.
    compute_sum()
        Compute the sum of all weighted WakeField into self.sum.
    plot_area(Z_type="Zlong", component="real", sigma=None, attr_list=None)
        Plot the contributions of different kind of WakeFields.
    save(file)
        Save impedance model to file.
    load(file)
        Load impedance model from file.
    """

    def __init__(self, ring):
        self.ring = ring
        self.optics = self.ring.optics
        self.wakefields = []
        self.positions = []
        self.names = []
        self.globals = []
        self.globals_names = []
        self.sum_names = []

    def add(self, wakefield, positions, name=None):
        """
        Add the same WakeField object at different locations to the model.

        Parameters
        ----------
        wakefield : WakeField
            WakeField object to add to the model.
        positions : array, float or int
            Array of longitudinal positions where the elements are loacted.
        name : str, optional
            Name of the element type. If None, the name of the WakeField object
            is used. The default is None.

        Returns
        -------
        None.

        """
        if name is None:
            name = wakefield.name
        if name is None:
            raise ValueError("Please give a valid name.")
        if name not in self.names:
            self.names.append(name)
        else:
            raise ValueError("This name is already taken.")
        self.wakefields.append(wakefield)
        self.positions.append(positions)

    def add_global(self, wakefield, name=None):
        """
        Add a "global" WakeField object which will added to the sum WakeField
        but not weighted by beta functions.

        To use with "distributed" elements, for example a resistive wall
        wakefield computed from an effective radius (so which has already been
        weighted by beta functions).

        Parameters
        ----------
        wakefield : WakeField
            WakeField object to add to the model.
        name : str, optional
            Name of the element type. If None, the name of the WakeField object
            is used. The default is None.

        Returns
        -------
        None.

        """
        self.globals.append(wakefield)
        if name is None:
            name = wakefield.name
        if name is None:
            raise ValueError("Please give a valid name.")
        if name not in self.globals_names:
            self.globals_names.append(name)
        else:
            raise ValueError("This name is already taken.")
        # setattr(self, name, wakefield)

    def sum_beta(self, wake, beta):
        """
        Weight a WakeField object by an array of beta functions.

        Parameters
        ----------
        wake : WakeField
            WakeField element object.
        beta : array of shape (2, N)
            Beta function at the locations of the elements.

        Returns
        -------
        wake_sum : WakeField
            WakeField object weighted by beta functions.

        """
        wake_sum = deepcopy(wake)
        local_beta = self.ring.optics.local_beta
        for component_name in wake_sum.components:
            comp = getattr(wake_sum, component_name)
            weight = ((beta[0, :] ** comp.power_x) *
                      (beta[1, :] ** comp.power_y))
            if comp.plane == "x":
                weight = weight.sum() / local_beta[0]
            if comp.plane == "y":
                weight = weight.sum() / local_beta[1]
            else:
                weight = weight.sum()
            setattr(wake_sum, component_name, weight*comp)
        return wake_sum

    def compute_sum_names(self):
        """
        Compute the weighted WakeField for each WakeField object type.
        The new summed WakeField object are set to into self.sum_name.
        """
        for idx, wake in enumerate(self.wakefields):
            attribute_name = "sum_" + self.names[idx]
            beta = self.optics.beta(self.positions[idx])
            wake_sum = self.sum_beta(wake, beta)
            wake_sum.name = attribute_name
            setattr(self, attribute_name, wake_sum)
            self.sum_names.append(attribute_name)

    def compute_sum(self):
        """
        Compute the sum of all weighted WakeField into self.sum.
        """
        self.compute_sum_names()
        for i, name in enumerate(self.sum_names):
            if i == 0:
                self.sum = deepcopy(getattr(self, name))
                self.sum.name = "sum"
            else:
                wake2 = getattr(self, name)
                for component_name2 in wake2.components:
                    comp2 = getattr(wake2, component_name2)
                    try:
                        comp1 = getattr(self.sum, component_name2)
                        setattr(self.sum, component_name2, comp1 + comp2)
                    except AttributeError:
                        setattr(self.sum, component_name2, comp2)
        for i, wake2 in enumerate(self.globals):
            name = self.globals_names[i]
            setattr(self, name, wake2)
            self.sum_names.append(name)
            if not hasattr(self, "sum"):
                self.sum = deepcopy(wake2)
                self.sum.name = "sum"
            else:
                for component_name2 in wake2.components:
                    comp2 = getattr(wake2, component_name2)
                    try:
                        comp1 = getattr(self.sum, component_name2)
                        setattr(self.sum, component_name2, comp1 + comp2)
                    except AttributeError:
                        setattr(self.sum, component_name2, comp2)

    def group_attributes(self, string_in_name, names_to_group=None, property_list=['Zlong']):
        """Groups attributes in the ImpedanceModel based on a given string pattern.
        Args:
            string_in_name (str): The string pattern used to match attribute names for grouping. If names_to_group is given, this is a name of a new attribute instead.
            names_to_group (list, optional): List of attribute names to be explicitly grouped.
                                            If not provided, attributes matching the string pattern will be automatically grouped.
                                            Defaults to None.
            property_list (list, optional): List of property names to be grouped for each attribute.
                                            Defaults to ['Zlong'].
        Returns:
            int: Returns 0 indicating the successful grouping of attributes.
        Notes:
            - This method groups attributes in the ImpedanceModel class based on a given string pattern.
            - If names_to_group is not provided, it automatically selects attributes from the existing sum_names that match the given string pattern.
            - A new WakeField instance is created and assigned to the attribute named 'string_in_name'.
            - The specified properties from the first attribute in names_to_group are appended to the new WakeField instance.
            - The values of the specified properties from the remaining attributes in names_to_group are added to the corresponding properties in the new WakeField instance.
            - The names_to_group attributes are removed from the ImpedanceModel class.
            - The new grouped attribute is added to sum_names, and the first attribute in names_to_group is removed from sum_names."""
        attrs = self.sum_names
        if names_to_group == None:
            names_to_group = []
            for attr in attrs:
                if string_in_name in attr:
                    names_to_group.append(attr)
        setattr(self, string_in_name, WakeField())
        for prop in property_list:
            getattr(self, string_in_name).append_to_model(
                getattr(getattr(self, names_to_group[0]), prop))
        for attr in names_to_group[1:]:
            for prop in property_list:
                old_values = getattr(getattr(self, string_in_name), prop)
                new_values = getattr(getattr(self, attr), prop)
                setattr(getattr(self, string_in_name),
                        prop, old_values+new_values)
            self.sum_names.remove(attr)
            delattr(self, attr)
        self.sum_names.append(string_in_name)
        self.sum_names.remove(names_to_group[0])
        delattr(self, names_to_group[0])
        return 0

    def rename_attribute(self, old_name, new_name):
        """Renames an attribute in the ImpedanceModel.
        Args:
            old_name (str): The current name of the attribute to be renamed.
            new_name (str): The new name for the attribute.
        Raises:
            KeyError: If the old_name doesn't exist as an attribute in the ImpedanceModel.
        Notes:
            - This method renames an attribute in the ImpedanceModel class.
            - The attribute with the old_name is removed from the class's dictionary (__dict__) using the pop() method.
            - The attribute is then added back to the class's dictionary with the new_name using the __dict__ attribute.
            - If the old_name doesn't exist as an attribute in the ImpedanceModel, a KeyError is raised.
        """
        self.__dict__[new_name] = self.__dict__.pop(old_name)

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
            attr_list = self.sum_names

        # manage legend
        Ztype_dict = {"Zlong": 0, "Zxdip": 1,
                      "Zydip": 2, "Zxquad": 3, "Zyquad": 4}
        scale = [1e-3, 1e-6, 1e-6, 1e-6, 1e-6]
        label_list = [r"$Z_\mathrm{long} \; (\mathrm{k}\Omega)$",
                      r"$\frac{1}{\beta_0} \sum_{j} \beta_{x,j} Z_{x,j}^\mathrm{Dip} \; (\mathrm{M}\Omega/\mathrm{m})$",
                      r"$\frac{1}{\beta_0} \sum_{j} \beta_{y,j} Z_{y,j}^\mathrm{Dip} \; (\mathrm{M}\Omega/\mathrm{m})$",
                      r"$\frac{1}{\beta_0} \sum_{j} \beta_{x,j} Z_{x,j}^\mathrm{Quad} \; (\mathrm{M}\Omega/\mathrm{m})$",
                      r"$\frac{1}{\beta_0} \sum_{j} \beta_{y,j} Z_{y,j}^\mathrm{Quad} \; (\mathrm{M}\Omega/\mathrm{m})$"]
        leg = Ztype_dict[Z_type]

        # sort plot by decresing area size
        area = np.zeros((len(attr_list),))
        for index, attr in enumerate(attr_list):
            try:
                sum_imp = getattr(getattr(self, attr), Z_type)
                area[index] = trapz(
                    sum_imp.data[component], sum_imp.data.index)
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
        colorblind = colormaps['tab10'].colors
        for index in sorted_index:
            attr = attr_list[index]
            # Set all impedances with common indexes using + zero_impedance
            try:
                sum_imp = getattr(getattr(self, attr), Z_type) + zero_impedance
                ax.fill_between(sum_imp.data.index*1e-9, total_imp,
                                total_imp + sum_imp.data[component]*scale[leg], edgecolor=colorblind[index % 10], color=colorblind[index % 10])
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

        ax.legend(legend, loc="upper left", ncol=2)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(label_list[leg] + " - " + component + " part")
        ax.set_title(label_list[leg] + " - " + component + " part")

        if zoom is True:
            in_ax = inset_axes(ax,
                               width="30%",  # width = 30% of parent_bbox
                               height=1.5,  # height : 1 inch
                               loc=1)

            total_imp = 0
            for index in sorted_index:
                attr = attr_list[index]
                # Set all impedances with common indexes using + zero_impedance
                try:
                    sum_imp = getattr(getattr(self, attr),
                                      Z_type) + zero_impedance
                    in_ax.fill_between(sum_imp.data.index*1e-3, total_imp,
                                       total_imp + sum_imp.data[component]*1e-9, edgecolor=colorblind[index % 10], color=colorblind[index % 10])
                    total_imp += sum_imp.data[component]*1e-9
                except AttributeError:
                    pass
            in_ax.set_xlim([0, 200])
            in_ax.set_xlabel("Frequency (kHz)")
            in_ax.set_ylabel(r"$[\mathrm{G}\Omega]$")

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

        eff_array = np.zeros((len(attr_list), 3), dtype=complex)

        for i, attr in enumerate(attr_list):
            try:
                impedance = getattr(getattr(self, attr), "Zlong")
                eff_array[i, 0] = effective_impedance(self.ring, impedance,
                                                      m, mu, sigma, M, tuneS,
                                                      xi, mode)
            except AttributeError:
                pass

            try:
                impedance = getattr(getattr(self, attr), "Zxdip")
                eff_array[i, 1] = effective_impedance(self.ring, impedance,
                                                      m, mu, sigma, M, tuneS,
                                                      xi, mode)
            except AttributeError:
                pass

            try:
                impedance = getattr(getattr(self, attr), "Zydip")
                eff_array[i, 2] = effective_impedance(self.ring, impedance,
                                                      m, mu, sigma, M, tuneS,
                                                      xi, mode)
            except AttributeError:
                pass

        eff_array[:, 0] = eff_array[:, 0]*self.ring.omega0*1e3
        eff_array[:, 1] = eff_array[:, 1]*1e-3
        eff_array[:, 2] = eff_array[:, 2]*1e-3

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

        beam_spect = beam_spectrum(f, M, bunch_spacing, sigma=sigma)

        bunch_spect = gaussian_bunch_spectrum(f, sigma)

        attr_list = self.sum_names

        loss_array = np.zeros((len(attr_list), 2))

        for i, attr in enumerate(attr_list):
            try:
                impedance = getattr(getattr(self, attr), "Zlong")
                loss_array[i, 0] = beam_loss_factor(
                    impedance, f, beam_spect, self.ring)
                loss_array[i, 1] = beam_loss_factor(
                    impedance, f, bunch_spect, self.ring)
            except AttributeError:
                pass

        loss_array = loss_array*1e-12
        summary = pd.DataFrame(loss_array, index=attr_list,
                               columns=["loss factor (beam) [V/pC]", "loss factor (bunch) [V/pC]"])

        summary["P (beam) [W]"] = summary["loss factor (beam) [V/pC]"] * \
            1e12*Q**2/(self.ring.T0)
        summary["P (bunch) [W]"] = summary["loss factor (bunch) [V/pC]"] * \
            1e12*Q**2/(self.ring.T0)*M

        return summary

    def power_loss_spectrum(self, sigma, M, bunch_spacing, I, n_points=10e6,
                            max_overlap=False, plot=False):
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
            double_sided_impedance(impedance)

        frequency = np.linspace(fmin, fmax, int(n_points))
        if max_overlap is False:
            spectrum = beam_spectrum(frequency, M, bunch_spacing, sigma)
        else:
            spectrum = gaussian_bunch_spectrum(frequency, sigma)*M

        pmax = np.floor(fmax/self.ring.f0)
        pmin = np.floor(fmin/self.ring.f0)

        p = np.arange(pmin+1, pmax)
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
            p1, = ax.plot(pf0, ReZ, color="r", label="Re[Z] [Ohm]")
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
        to_save = {"wakefields": self.wakefields,
                   "positions": self.positions,
                   "names": self.names,
                   "globals": self.globals,
                   "globals_names": self.globals_names}
        with open(file, "wb") as f:
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
        self.names = to_load["names"]
        self.globals = to_load["globals"]
        self.globals_names = to_load["globals_names"]
        self.compute_sum()
