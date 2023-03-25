# -*- coding: utf-8 -*-
"""
This module defines general classes to describes wakefields, impedances and 
wake functions.
"""

import warnings
import pickle
import pandas as pd
import numpy as np
import scipy as sc
from copy import deepcopy
from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.constants import c

class ComplexData:
    """
    Define a general data structure for a complex function based on a pandas 
    DataFrame.

    Parameters
    ----------
    variable : list, numpy array
       contains the function variable values

    function : list or numpy array of comp lex numbers
        contains the values taken by the complex function
    """

    def __init__(self, variable=np.array([-1e15, 1e15]),
                 function=np.array([0, 0])):
        self.data = pd.DataFrame({'real': np.real(function),
                                  'imag': np.imag(function)},
                                 index=variable)
        self.data.index.name = 'variable'

    def add(self, structure_to_add, method='zero', interp_kind='cubic', 
            index_name="variable"):
        """
        Method to add two structures. If the data don't have the same length,
        different cases are handled.
        
        Parameters
        ----------
        structure_to_add : ComplexData object, int, float or complex.
            structure to be added.
        method : str, optional
            manage how the addition is done, possibilties are: 
            -common: the common range of the index (variable) from the two
                structures is used. The input data are cross-interpolated
                so that the result data contains the points at all initial
                points in the common domain.
            -extrapolate: extrapolate the value of both ComplexData.
            -zero: outside of the common range, the missing values are zero. In 
                the common range, behave as "common" method.
        interp_kind : str, optional
            interpolation method which is passed to pandas and to
            scipy.interpolate.interp1d.
        index_name : str, optional
            name of the dataframe index passed to the method
            
        Returns
        -------
        ComplexData 
            Contains the sum of the two inputs.
        """

        # Create first two new DataFrames merging the variable
        # from the two input data

        if isinstance(structure_to_add, (int, float, complex)):
            structure_to_add = ComplexData(variable=self.data.index,
                                           function=(structure_to_add * 
                                                     np.ones(len(self.data.index))))
                                
        data_to_concat = structure_to_add.data.index.to_frame().set_index(index_name)
        
        initial_data = pd.concat([self.data, data_to_concat], sort=True)
        initial_data = initial_data[~initial_data.index.duplicated(keep='first')]
        initial_data = initial_data.sort_index()

        data_to_add = pd.concat(
                        [structure_to_add.data,
                         self.data.index.to_frame().set_index(index_name)],
                        sort=True)
        data_to_add = data_to_add[~data_to_add.index.duplicated(keep='first')]
        data_to_add = data_to_add.sort_index()

        if method == 'common':
            max_variable = min(structure_to_add.data.index.max(),
                               self.data.index.max())

            min_variable = max(structure_to_add.data.index.min(),
                               self.data.index.min())

            initial_data = initial_data.interpolate(method=interp_kind)
            mask = ((initial_data.index <= max_variable)
                    & (initial_data.index >= min_variable))
            initial_data = initial_data[mask]

            data_to_add = data_to_add.interpolate(method=interp_kind)
            mask = ((data_to_add.index <= max_variable)
                    & (data_to_add.index >= min_variable))
            data_to_add = data_to_add[mask]

            result_structure = ComplexData()
            result_structure.data = initial_data + data_to_add
            return result_structure

        if method == 'extrapolate':
            print('Not there yet')
            return self
        
        if method == 'zero':
            max_variable = min(structure_to_add.data.index.max(),
                               self.data.index.max())

            min_variable = max(structure_to_add.data.index.min(),
                               self.data.index.min())

            mask = ((initial_data.index <= max_variable)
                    & (initial_data.index >= min_variable))
            initial_data[mask] = initial_data[mask].interpolate(method=interp_kind)

            mask = ((data_to_add.index <= max_variable)
                    & (data_to_add.index >= min_variable))
            data_to_add[mask] = data_to_add[mask].interpolate(method=interp_kind)
            
            initial_data.replace(to_replace=np.nan, value=0, inplace=True)
            data_to_add.replace(to_replace=np.nan, value=0, inplace=True)
            
            result_structure = ComplexData()
            result_structure.data = initial_data + data_to_add
            return result_structure

    def __radd__(self, structure_to_add):
        return self.add(structure_to_add, method='zero')

    def __add__(self, structure_to_add):
        return self.add(structure_to_add, method='zero')

    def multiply(self, factor):
        """
        Multiply a data strucure with a float or an int.
        If the multiplication is done with something else, throw a warning.
        """
        if isinstance(factor, (int, float)):
            result_structure = ComplexData()
            result_structure.data = self.data * factor
            return result_structure
        else:
            warnings.warn(('The multiplication factor is not a float '
                           'or an int.'), UserWarning)
            return self

    def __mul__(self, factor):
        return self.multiply(factor)

    def __rmul__(self, factor):
        return self.multiply(factor)
    
    def __call__(self, values, interp_kind="cubic"):
        """
        Interpolation of the data by calling the class to have a function-like
        behaviour.
        
        Parameters
        ----------
        values : list or numpy array of complex, int or float
            values to be interpolated.
        interp_kind : str, optional
            interpolation method which is passed to scipy.interpolate.interp1d.
            
        Returns
        -------
        numpy array 
            Contains the interpolated data.
        """
        real_func = interp1d(x = self.data.index, 
                             y = self.data["real"], kind=interp_kind)
        imag_func = interp1d(x = self.data.index, 
                             y = self.data["imag"], kind=interp_kind)
        return real_func(values) + 1j*imag_func(values)
    
    
    def initialize_coefficient(self):
        """
        Define the impedance coefficients and the plane of the impedance that
        are presents in attributes of the class.
        """
        table = self.name_and_coefficients_table()
        
        try:
            component_coefficients = table[self.component_type].to_dict()
        except KeyError:
            print('Component type {} does not exist'.format(self.component_type))
            raise
        
        self.a = component_coefficients["a"]
        self.b = component_coefficients["b"]
        self.c = component_coefficients["c"]
        self.d = component_coefficients["d"]
        self.plane = component_coefficients["plane"]
                    
    def name_and_coefficients_table(self):
        """
        Return a table associating the human readbale names of an impedance
        component and its associated coefficients and plane.
        """

        component_dict = {
            'long': {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'plane': 'z'},
            'xcst': {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'plane': 'x'},
            'ycst': {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'plane': 'y'},
            'xdip': {'a': 1, 'b': 0, 'c': 0, 'd': 0, 'plane': 'x'},
            'ydip': {'a': 0, 'b': 1, 'c': 0, 'd': 0, 'plane': 'y'},
            'xydip': {'a': 0, 'b': 1, 'c': 0, 'd': 0, 'plane': 'x'},
            'yxdip': {'a': 1, 'b': 0, 'c': 0, 'd': 0, 'plane': 'y'},
            'xquad': {'a': 0, 'b': 0, 'c': 1, 'd': 0, 'plane': 'x'},
            'yquad': {'a': 0, 'b': 0, 'c': 0, 'd': 1, 'plane': 'y'},
            'xyquad': {'a': 0, 'b': 0, 'c': 0, 'd': 1, 'plane': 'x'},
            'yxquad': {'a': 0, 'b': 0, 'c': 1, 'd': 0, 'plane': 'y'},
            }

        return pd.DataFrame(component_dict)
    
    @property
    def power_x(self):
        power_x = self.a/2 + self.c/2.
        if self.plane == 'x':
            power_x += 1./2.
        return power_x

    @property
    def power_y(self):
        power_y = self.b/2. + self.d/2.
        if self.plane == 'y':
            power_y += 1./2.
        return power_y
    
    @property
    def component_type(self):
        return self._component_type
    
    @component_type.setter
    def component_type(self, value):
        self._component_type = value
        self.initialize_coefficient()
    
class WakeFunction(ComplexData):
    """
    Define a WakeFunction object based on a ComplexData object.
    
    Parameters
    ----------
    variable : array-like
        Time domain structure of the wake function in [s].
    function : array-like
        Wake function values in [V/C].
    component_type : str, optinal
        Type of the wake function: "long", "xdip", "xquad", ...
    
    Attributes
    ----------
    data : DataFrame
    wake_type : str
    
    Methods
    -------
    from_wakepotential(file_name, bunch_length, bunch_charge, freq_lim)
        Compute a wake function from a wake potential file and load it to the 
        WakeFunction object.
    """

    def __init__(self,
                 variable=np.array([-1e15, 1e15]),
                 function=np.array([0, 0]), component_type='long'):
        super().__init__(variable, function)
        self._component_type = component_type
        self.data.index.name = "time [s]"
        self.initialize_coefficient()
        
    def add(self, structure_to_add, method='zero'):
        """
        Method to add two WakeFunction objects. The two structures are
        compared so that the addition is self-consistent.
        """
 
        if isinstance(structure_to_add, (int, float, complex)):
            result = super().add(structure_to_add, method=method,
                                 index_name="time [s]")
        elif isinstance(structure_to_add, WakeFunction):
            if (self.component_type == structure_to_add.component_type):
                result = super().add(structure_to_add, method=method,
                                     index_name="time [s]")
            else:
                warnings.warn(('The two WakeFunction objects do not have the '
                               'same coordinates or plane or type. '
                               'Returning initial WakeFunction object.'),
                              UserWarning)
                result = self
               
        wake_to_return = WakeFunction(
                                result.data.index,
                                result.data.real.values,
                                self.component_type)   
        return wake_to_return
 
    def __radd__(self, structure_to_add):
        return self.add(structure_to_add)
 
    def __add__(self, structure_to_add):
        return self.add(structure_to_add)
 
    def multiply(self, factor):
        """
        Multiply a WakeFunction object with a float or an int.
        If the multiplication is done with something else, throw a warning.
        """
        result = super().multiply(factor)
        wake_to_return = WakeFunction(
                                result.data.index,
                                result.data.real.values,
                                self.component_type)   
        return wake_to_return
 
    def __mul__(self, factor):
        return self.multiply(factor)
 
    def __rmul__(self, factor):
        return self.multiply(factor)
        
    def from_wakepotential(self, file_name, bunch_length, bunch_charge, 
                           freq_lim, component_type="long", divide_by=None, 
                           nout=None, trim=False, axis0='dist', 
                           axis0_scale=1e-3, axis1_scale=1e-12):
        """
        Compute a wake function from a wake potential file and load it to the 
        WakeFunction object.
    
        Parameters
        ----------
        file_name : str
            Text file that contains wake potential data.
        bunch_length : float
            Spatial bunch length in [m].
        bunch_charge : float
            Bunch charge in [C].
        freq_lim : float
            The maximum frequency for calculating the impedance [Hz].
        component_type : str, optional
            Type of the wake: "long", "xdip", "xquad", ...
        divide_by : float, optional
            Divide the wake potential by a value. Mainly used to normalize 
            transverse wake function by displacement.
        nout : int, optional
            Length of the transformed axis of the output. If None, it is taken
            to be 2*(n-1) where n is the length of the input. If nout > n, the
            input is padded with zeros. If nout < n, the inpu it cropped.
            Note that, for any nout value, nout//2+1 input points are required.
        trim : bool or float, optional
            If True, the pseudo wake function is trimmed at time=0 and is forced 
            to zero where time<0. 
            If False, the original result is preserved.
            If a float is given, the pseudo wake function is trimmed from 
            time <= trim to 0.
        axis0 : {'dist', 'time'}, optional
            Viariable of the data file's first column. Use 'dist' for spacial 
            distance. Otherwise, 'time' for temporal distance. 
        axis0_scale : float, optional
            Scale of the first column with respect to meter if axis0 = 'dist',
            or to second if axis0 = 'time'.
        axis1_scale : float, optional
            Scale of the wake potential (in second column) with respect to V/C.
    
        """
        
        imp = Impedance()
        imp.from_wakepotential(file_name=file_name, bunch_length=bunch_length,
                               bunch_charge=bunch_charge, freq_lim=freq_lim,
                               component_type=component_type, divide_by=divide_by,
                               axis0=axis0, axis0_scale=axis0_scale,
                               axis1_scale=axis1_scale)
        wf = imp.to_wakefunction(nout=nout, trim=trim)
        self.__init__(variable=wf.data.index, function=wf.data["real"],
                      component_type=component_type)
        
class Impedance(ComplexData):
    """
    Define an Impedance object based on a ComplexData object.
    
    Parameters
    ----------
    variable : array-like
        Frequency domain structure of the impedance in [Hz].
    function : array-like
        Impedance values in [Ohm].
    component_type : str, optinal
        Type of the impedance: "long", "xdip", "xquad", ...
    
    Attributes
    ----------
    data : DataFrame
    impedance_type : str
    
    Methods
    -------
    from_wakepotential(file_name, bunch_length, bunch_charge, freq_lim)
        Compute impedance from wake potential data and load it to the Impedance
        object.
    loss_factor(sigma)
        Compute the loss factor or the kick factor for a Gaussian bunch.
    to_wakefunction()
        Return a WakeFunction object from the impedance data.
    """

    def __init__(self,
                 variable=np.array([-1e15, 1e15]),
                 function=np.array([0, 0]), component_type='long'):
        super().__init__(variable, function)
        self._component_type = component_type
        self.data.index.name = 'frequency [Hz]'
        self.initialize_coefficient()

    def add(self, structure_to_add, beta_x=1, beta_y=1, method='zero'):
        """
        Method to add two Impedance objects. The two structures are
        compared so that the addition is self-consistent.
        Beta functions can be precised as well.
        """

        if isinstance(structure_to_add, (int, float, complex)):
            result = super().add(structure_to_add, method=method, 
                                 index_name="frequency [Hz]")
        elif isinstance(structure_to_add, Impedance):
            if (self.component_type == structure_to_add.component_type):
                weight = (beta_x ** self.power_x) * (beta_y ** self.power_y)
                result = super().add(structure_to_add * weight, method=method, 
                                     index_name="frequency [Hz]")
            else:
                warnings.warn(('The two Impedance objects do not have the '
                               'same coordinates or plane or type. '
                               'Returning initial Impedance object.'),
                              UserWarning)
                result = self
                
        impedance_to_return = Impedance(
                                result.data.index,
                                result.data.real.values + 1j*result.data.imag.values,
                                self.component_type)    
        return impedance_to_return

    def __radd__(self, structure_to_add):
        return self.add(structure_to_add)

    def __add__(self, structure_to_add):
        return self.add(structure_to_add)

    def multiply(self, factor):
        """
        Multiply a Impedance object with a float or an int.
        If the multiplication is done with something else, throw a warning.
        """
        result = super().multiply(factor)
        impedance_to_return = Impedance(
                                result.data.index,
                                result.data.real.values + 1j*result.data.imag.values,
                                self.component_type)    
        return impedance_to_return

    def __mul__(self, factor):
        return self.multiply(factor)

    def __rmul__(self, factor):
        return self.multiply(factor)
    
    def loss_factor(self, sigma):
        """
        Compute the loss factor or the kick factor for a Gaussian bunch. 
        Formulas from Eq. (2) p239 and Eq.(7) p240 of [1].
        
        Parameters
        ----------
        sigma : float
            RMS bunch length in [s]
        
        Returns
        -------
        kloss: float
            Loss factor in [V/C] or kick factor in [V/C/m] depanding on the 
            impedance type.
            
        References
        ----------
        [1] : Handbook of accelerator physics and engineering, 3rd printing.
        """
        
        positive_index = self.data.index > 0
        frequency = self.data.index[positive_index]
        
        # Import here to avoid circular import
        from mbtrack2.utilities import spectral_density
        sd = spectral_density(frequency, sigma, m=0)
        
        if(self.component_type == "long"):
            kloss = trapz(x = frequency, 
                          y = 2*self.data["real"][positive_index]*sd)
        elif(self.component_type == "xdip" or self.component_type == "ydip"):
            kloss = trapz(x = frequency, 
                          y = 2*self.data["imag"][positive_index]*sd)
        elif(self.component_type == "xquad" or self.component_type == "yquad"):
            kloss = trapz(x = frequency, 
                          y = 2*self.data["imag"][positive_index]*sd)
        else:
            raise TypeError("Impedance type not recognized.")

        return kloss
    
    def from_wakepotential(self, file_name, bunch_length, bunch_charge, 
                           freq_lim, component_type="long", divide_by=None,
                           axis0='dist', axis0_scale=1e-3, axis1_scale=1e-12):
        """
        Compute impedance from wake potential data and load it to the Impedance
        object.

        Parameters
        ----------
        file_name : str
            Text file that contains wake potential data.
        bunch_length : float
            Electron bunch lenth [m]. 
        bunch_charge : float
            Total bunch charge [C].
        freq_lim : float
            The maximum frequency for calculating the impedance [Hz].
        component_type : str, optional
            Type of the impedance: "long", "xdip", "xquad", ...
        divide_by : float, optional
            Divide the impedance by a value. Mainly used to normalize transverse 
            impedance by displacement.
        axis0 : {'dist', 'time'}, optional
            Viariable of the data file's first column. Use 'dist' for spacial 
            distance. Otherwise, 'time' for temporal distance. 
        axis0_scale : float, optional
            Scale of the first column with respect to meter if axis0 = 'dist',
            or to second if axis0 = 'time'.
        axis1_scale : float, optional
            Scale of the wake potential (in second column) with respect to V/C.

        """
        
        s, wp0 = np.loadtxt(file_name, unpack=True)
        if axis0 == 'dist':
            tau = s*axis0_scale/c
        elif axis0 == 'time':
            tau = s*axis0_scale
        else:
            raise TypeError('Type of axis 0 not recognized.')
            
        wp = wp0 / axis1_scale
        if divide_by is not None:
            wp = wp / divide_by
        
        # FT of wake potential
        sampling = tau[1] - tau[0]
        freq = sc.fft.rfftfreq(len(tau), sampling)
        dtau = (tau[-1]-tau[0])/len(tau)
        dft_wake = sc.fft.rfft(wp) * dtau
        
        # FT of analytical bunch profile and analytical impedance
        sigma = bunch_length/c
        mu = tau[0]
        
        i_limit = freq < freq_lim
        freq_trun = freq[i_limit]
        dft_wake_trun = dft_wake[i_limit]
        
        dft_rho_trun = np.exp(-0.5*(sigma*2*np.pi*freq_trun)**2 + \
                                  1j*mu*2*np.pi*freq_trun)*bunch_charge
        if component_type == "long":
            imp = dft_wake_trun/dft_rho_trun*-1*bunch_charge
        elif (component_type == "xdip") or (component_type == "ydip"):
            imp = dft_wake_trun/dft_rho_trun*-1j*bunch_charge
        else:
            raise NotImplementedError(component_type + " is not correct.")
            
        self.__init__(variable=freq_trun, function=imp, 
                         component_type=component_type)
        
    def to_wakefunction(self, nout=None, trim=False):
        """
        Return a WakeFunction object from the impedance data.
        The impedance data is assumed to be sampled equally.
    
        Parameters
        ----------
        nout : int or float, optional
            Length of the transformed axis of the output. If None, it is taken
            to be 2*(n-1) where n is the length of the input. If nout > n, the
            input is padded with zeros. If nout < n, the input it cropped.
            Note that, for any nout value, nout//2+1 input points are required.
        trim : bool or float, optional
            If True, the pseudo wake function is trimmed at time=0 and is forced 
            to zero where time<0. 
            If False, the original result is preserved.
            If a float is given, the pseudo wake function is trimmed from 
            time <= trim to 0. 
        """
        
        Z0 = (self.data['real'] + self.data['imag']*1j)
        Z = Z0[~np.isnan(Z0)]
        
        if self.component_type != "long":
            Z = Z / 1j
        
        freq = Z.index
        fs = ( freq[-1] - freq[0] ) / len(freq)
        sampling = freq[1] - freq[0]
        
        if nout is None:
            nout = len(Z)
        else:
            nout = int(nout)
            
        time_array = sc.fft.fftfreq(nout, sampling)
        Wlong_raw = sc.fft.irfft(np.array(Z), n=nout, axis=0) * nout * fs
        
        time = sc.fft.fftshift(time_array)
        Wlong = sc.fft.fftshift(Wlong_raw)
        
        if trim is not False:
            i_neg = np.where(time<trim)[0]
            Wlong[i_neg] = 0
                    
        wf = WakeFunction(variable=time, function=Wlong, 
                          component_type=self.component_type)
        return wf
    
class WakeField:
    """
    Defines a WakeField which corresponds to a single physical element which 
    produces different types of wakes, represented by their Impedance or 
    WakeFunction objects.
    
    Parameters
    ----------
    structure_list : list, optional
        list of Impedance/WakeFunction objects to add to the WakeField.
    name : str, optional
    
    Attributes
    ----------
    impedance_components : array of str
        Impedance components present for this element.
    wake_components : array of str
        WakeFunction components present for this element.
    components : array of str
        Impedance or WakeFunction components present for this element.
        
    Methods
    -------
    append_to_model(structure_to_add)
        Add Impedance/WakeFunction to WakeField.
    list_to_attr(structure_list)
        Add list of Impedance/WakeFunction to WakeField.
    add_wakefields(wake1, beta1, wake2, beta2)
        Add two WakeFields taking into account beta functions.
    add_several_wakefields(wakefields, beta)
        Add a list of WakeFields taking into account beta functions.
    drop(component)
        Delete a component or a list of component from the WakeField.
    save(file)
        Save WakeField element to file.
    load(file)
        Load WakeField element from file.
    """

    def __init__(self, structure_list=None, name=None):
        self.list_to_attr(structure_list)
        self.name = name

    def append_to_model(self, structure_to_add):
        """
        Add Impedance/WakeFunction component to WakeField.

        Parameters
        ----------
        structure_to_add : Impedance or WakeFunction object
        """
        list_of_attributes = dir(self)
        if isinstance(structure_to_add, Impedance):
            attribute_name = "Z" + structure_to_add.component_type
            if attribute_name in list_of_attributes:
                raise ValueError("There is already a component of the type "
                                 "{} in this element.".format(attribute_name))
            else:
                self.__setattr__(attribute_name, structure_to_add)
        elif isinstance(structure_to_add, WakeFunction):
            attribute_name = "W" + structure_to_add.component_type
            if attribute_name in list_of_attributes:
                raise ValueError("There is already a component of the type "
                                 "{} in this element.".format(attribute_name))
            else:
                self.__setattr__(attribute_name, structure_to_add)
        else:
            raise ValueError("{} is not an Impedance nor a WakeFunction.".format(structure_to_add))
    
    def list_to_attr(self, structure_list):
        """
         Add list of Impedance/WakeFunction components to WakeField.

        Parameters
        ----------
        structure_list : list of Impedance or WakeFunction objects.
        """
        if structure_list is not None:
            for component in structure_list:
                self.append_to_model(component)
    
    @property
    def impedance_components(self):
        """
        Return an array of the impedance component names for the element.
        """
        valid = ["Zlong", "Zxdip", "Zydip", "Zxquad", "Zyquad"]
        return np.array([comp for comp in dir(self) if comp in valid])
    
    @property
    def wake_components(self):
        """
        Return an array of the wake function component names for the element.
        """
        valid = ["Wlong", "Wxdip", "Wydip", "Wxquad", "Wyquad"]
        return np.array([comp for comp in dir(self) if comp in valid])
    
    @property
    def components(self):
        """
        Return an array of the component names for the element.
        """
        valid = ["Wlong", "Wxdip", "Wydip", "Wxquad", "Wyquad", 
                 "Zlong", "Zxdip", "Zydip", "Zxquad", "Zyquad"]
        return np.array([comp for comp in dir(self) if comp in valid])
    
    def drop(self, component):
        """
        Delete a component or a list of component from the WakeField.

        Parameters
        ----------
        component : str or list of str
            Component or list of components to drop.
            If "Z", drop all impedance components.
            If "W"", drop all wake function components.

        Returns
        -------
        None.

        """
        if component == "Z":
            component = self.impedance_components
        elif component == "W":
            component = self.wake_components
        
        if isinstance(component, str):
            delattr(self, component)
        elif isinstance(component, list) or isinstance(component, np.ndarray):
            for comp in component:
                delattr(self, comp)
        else:
            raise TypeError("component should be a str or list of str.")
            
    def save(self, file):
        """
        Save WakeField element to file.
        
        The same pandas version is needed on both saving and loading computer
        for the pickle to work.

        Parameters
        ----------
        file : str
            File where the WakeField element is saved.

        Returns
        -------
        None.

        """
        with open(file,"wb") as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(file):
        """
        Load WakeField element from file.
        
        The same pandas version is needed on both saving and loading computer
        for the pickle to work.

        Parameters
        ----------
        file : str
            File where the WakeField element is saved.

        Returns
        -------
        wakefield : WakeField
            Loaded wakefield.

        """
        with open(file, 'rb') as f:
            wakefield = pickle.load(f)
            
        return wakefield
    
    @staticmethod
    def add_wakefields(wake1, beta1, wake2, beta2):
        """
        Add two WakeFields taking into account beta functions.

        Parameters
        ----------
        wake1 : WakeField
            WakeField to add.
        beta1 : array of shape (2,)
            Beta functions at wake1 postion.
        wake2 : WakeField
            WakeField to add.
        beta2 : array of shape (2,)
            Beta functions at wake2 postion.

        Returns
        -------
        wake_sum : WakeField
            WakeField with summed components.

        """
        wake_sum = deepcopy(wake1)
        for component_name1 in wake1.components:
            comp1 = getattr(wake_sum, component_name1)
            weight1 = ((beta1[0] ** comp1.power_x) * 
                      (beta1[1] ** comp1.power_y))
            setattr(wake_sum, component_name1, weight1*comp1)
            
        for component_name2 in wake2.components: 
            comp2 = getattr(wake2, component_name2)
            weight2 = ((beta2[0] ** comp2.power_x) * 
                      (beta2[1] ** comp2.power_y))
            try:
                comp1 = getattr(wake_sum, component_name2)
                setattr(wake_sum, component_name2, comp1 +
                        weight2*comp2)
            except AttributeError:
                setattr(wake_sum, component_name2, weight2*comp2)

        return wake_sum
    
    @staticmethod
    def add_several_wakefields(wakefields, beta):
        """
        Add a list of WakeFields taking into account beta functions.
        
        Parameters
        ----------
        wakefields : list of WakeField
            WakeFields to add.
        beta : array of shape (len(wakefields), 2)
            Beta functions at the WakeField postions.

        Returns
        -------
        wake_sum : WakeField
            WakeField with summed components..

        """
        if len(wakefields) == 1:
            return wakefields[0]
        elif len(wakefields) > 1:
            wake_sum = WakeField.add_wakefields(wakefields[0], beta[:,0],
                                     wakefields[1], beta[:,1])
            for i in range(len(wakefields) - 2):
                wake_sum = WakeField.add_wakefields(wake_sum, [1 ,1], 
                                         wakefields[i+2], beta[:,i+2])
            return wake_sum
        else:
            raise ValueError("Error in input.")
        