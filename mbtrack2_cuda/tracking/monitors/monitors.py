# -*- coding: utf-8 -*-
"""
This module defines the different monitor class which are used to save data
during tracking.
"""

import numpy as np
import h5py as hp
import random
from mbtrack2_cuda.tracking.element import Element
from mbtrack2_cuda.tracking.particles import Bunch, Beam
from mbtrack2_cuda.tracking.rf import CavityResonator
from scipy.interpolate import interp1d
from abc import ABCMeta
from scipy.fft import rfft, rfftfreq

class Monitor(Element, metaclass=ABCMeta):
    """
    Abstract Monitor class used for subclass inheritance to define all the
    different kind of monitors objects. 
    
    The Monitor class is based on h5py module to be able to write data on 
    structured binary files. The class provides a common file where the 
    different Monitor subclass can write.
    
    Attributes
    ----------
    file : HDF5 file
        Common file where all monitors, Monitor subclass elements, write the
        saved data. Based on class attribute _file_storage.
    file_name : string
        Name of the HDF5 file where the data is stored. Based on class 
        attribute _file_name_storage.
        
    Methods
    -------
    monitor_init(group_name, save_every, buffer_size, total_size,
                     dict_buffer, dict_file, file_name=None, mpi_mode=True)
        Method called to initialize Monitor subclass.
    write()
        Write data from buffer to the HDF5 file.
    to_buffer(object_to_save)
        Save data to buffer.
    close()
        Close the HDF5 file shared by all Monitor subclass, must be called 
        by at least an instance of a Montior subclass at the end of the 
        tracking.
    track_bunch_data(object_to_save)
        Track method to use when saving bunch data.
    """
    
    _file_name_storage = []
    _file_storage = []
    
    @property
    def file_name(self):
        """Common file where all monitors, Monitor subclass elements, write the
        saved data."""
        try:
            return self._file_name_storage[0]
        except IndexError:
            print("The HDF5 file name for monitors is not set.")
            raise ValueError
            
    @property
    def file(self):
        """Name of the HDF5 file where the data is stored."""
        try:
            return self._file_storage[0]
        except IndexError:
            print("The HDF5 file to store data is not set.")
            raise ValueError
            
    def monitor_init(self, group_name, save_every, buffer_size, total_size,
                     dict_buffer, dict_file, file_name=None, mpi_mode=False,
                     dict_dtype=None):
        """
        Method called to initialize Monitor subclass. 
        
        Parameters
        ----------
        group_name : string
            Name of the HDF5 group in which the data for the current monitor 
            will be saved.
        save_every : int or float
            Set the frequency of the save. The data is saved every save_every 
            call of the montior.
        buffer_size : int or float
            Size of the save buffer.
        total_size : int or float
            Total size of the save. The following relationships between the 
            parameters must exist: 
                total_size % buffer_size == 0
                number of call to track / save_every == total_size
        dict_buffer : dict
            Dictionary with keys as the attribute name to save and values as
            the shape of the buffer to create to hold the attribute, like 
            (key.shape, buffer_size)
        dict_file : dict
            Dictionary with keys as the attribute name to save and values as
            the shape of the dataset to create to hold the attribute, like 
            (key.shape, total_size)
        file_name : string, optional
            Name of the HDF5 where the data will be stored. Must be specified
            the first time a subclass of Monitor is instancied and must be None
            the following times.
        mpi_mode : bool, optional
            If True, open the HDF5 file in parallel mode, which is needed to
            allow several cores to write in the same file at the same time.
            If False, open the HDF5 file in standard mode.
        dict_dtype : dict, optional
            Dictionary with keys as the attribute name to save and values as
            the dtype to use to save the values.
            If None, float is used for all attributes.
        """
        
        # setup and open common file for all monitors
        if file_name is not None:
            if len(self._file_name_storage) == 0:
                self._file_name_storage.append(file_name + ".hdf5")
                if len(self._file_storage) == 0:
                    if mpi_mode == True:
                        from mpi4py import MPI
                        f = hp.File(self.file_name, "a", libver='earliest', 
                             driver='mpio', comm=MPI.COMM_WORLD)
                    else:
                        f = hp.File(self.file_name, "a", libver='earliest')
                    self._file_storage.append(f)
                else:
                    raise ValueError("File is already open.")
            else:
                raise ValueError("File name for monitors is already attributed.")
        
        self.group_name = group_name
        self.save_every = int(save_every)
        self.total_size = int(total_size)
        self.buffer_size = int(buffer_size)
        if total_size % buffer_size != 0:
            raise ValueError("total_size must be divisible by buffer_size.")
        self.buffer_count = 0
        self.write_count = 0
        self.track_count = 0
        
        # setup attribute buffers from values given in dict_buffer
        for key, value in dict_buffer.items():
            if dict_dtype == None:
                self.__setattr__(key,np.zeros(value))
            else:
                self.__setattr__(key,np.zeros(value, dtype=dict_dtype[key]))
        self.time = np.zeros((self.buffer_size,), dtype=int)

        # create HDF5 groups and datasets to save data from group_name and 
        # dict_file
        self.g = self.file.require_group(self.group_name)
        self.g.require_dataset("time", (self.total_size,), dtype=int)
        for key, value in dict_file.items():
            if dict_dtype == None:
                self.g.require_dataset(key, value, dtype=float)
            else:
                self.g.require_dataset(key, value, dtype=dict_dtype[key])
        
        # create a dictionary which handle slices
        slice_dict = {}
        for key, value in dict_file.items():
            slice_dict[key] = []
            for i in range(len(value)-1):
                slice_dict[key].append(slice(None))
        self.slice_dict = slice_dict
        
    def write(self):
        """Write data from buffer to the HDF5 file."""
        
        self.file[self.group_name]["time"][self.write_count*self.buffer_size:(
                    self.write_count+1)*self.buffer_size] = self.time
        for key, value in self.dict_buffer.items():
            slice_list = list(self.slice_dict[key])
            slice_list.append(slice(self.write_count*self.buffer_size,
                                    (self.write_count+1)*self.buffer_size))
            slice_tuple = tuple(slice_list)
            self.file[self.group_name][key][slice_tuple] = self.__getattribute__(key)
        
        self.file.flush()
        self.write_count += 1
        
    def to_buffer(self, object_to_save):
        """
        Save data to buffer.
        
        Parameters
        ----------
        object_to_save : python object
            Depends on the Monitor subclass, typically a Beam or Bunch object.
        """
        self.time[self.buffer_count] = self.track_count
        for key, value in self.dict_buffer.items():
            slice_list = list(self.slice_dict[key])
            slice_list.append(self.buffer_count)
            slice_tuple = tuple(slice_list)
            self.__getattribute__(key)[slice_tuple] = object_to_save.__getattribute__(key)
        self.buffer_count += 1
        
        if self.buffer_count == self.buffer_size:
            self.write()
            self.buffer_count = 0
            
    def close(self):
        """
        Close the HDF5 file shared by all Monitor subclass, must be called 
        by at least an instance of a Montior subclass at the end of the 
        tracking.
        """
        try:
            self.file.close()
            Monitor._file_name_storage = []
            Monitor._file_storage = []
        except ValueError:
            pass
        
    def track_bunch_data(self, object_to_save, check_empty=False):
        """
        Track method to use when saving bunch data.
        
        Parameters
        ----------
        object_to_save : Beam or Bunch
        check_emptiness: bool
            If True, check if the bunch is empty. If it is, then do nothing.
        """
        save = True
        if self.track_count % self.save_every == 0:
            
            if isinstance(object_to_save, Beam):
                if (object_to_save.mpi_switch == True):
                    if object_to_save.mpi.bunch_num == self.bunch_number:
                        bunch = object_to_save[object_to_save.mpi.bunch_num]
                    else:
                        save = False
                else:
                    bunch = object_to_save[self.bunch_number]
            elif isinstance(object_to_save, Bunch):
                bunch = object_to_save
            else:                            
                raise TypeError("object_to_save should be a Beam or Bunch object.")
            
            if save:
                if (check_empty == False) or (bunch.is_empty == False):
                    self.to_buffer(bunch)
                
        self.track_count += 1
            
class BunchMonitor(Monitor):
    """
    Monitor a single bunch and save attributes 
    (mean, std, emit, current, and cs_invariant).
    
    Parameters
    ----------
    bunch_number : int
        Bunch to monitor
    save_every : int or float
        Set the frequency of the save. The data is saved every save_every 
        call of the montior.
    buffer_size : int or float
        Size of the save buffer.
    total_size : int or float
        Total size of the save. The following relationships between the 
        parameters must exist: 
            total_size % buffer_size == 0
            number of call to track / save_every == total_size
    file_name : string, optional
        Name of the HDF5 where the data will be stored. Must be specified
        the first time a subclass of Monitor is instancied and must be None
        the following times.
    mpi_mode : bool, optional
        If True, open the HDF5 file in parallel mode, which is needed to
        allow several cores to write in the same file at the same time.
        If False, open the HDF5 file in standard mode.

    Methods
    -------
    track(object_to_save)
        Save data
    """
    
    def __init__(self, bunch_number, save_every, buffer_size, total_size, 
                 file_name=None, mpi_mode=False):
        
        self.bunch_number = bunch_number
        group_name = "BunchData_" + str(self.bunch_number)
        dict_buffer = {"mean":(6, buffer_size), "std":(6, buffer_size),
                     "emit":(3, buffer_size), "current":(buffer_size,),
                     "cs_invariant":(2, buffer_size)}
        dict_file = {"mean":(6, total_size), "std":(6, total_size),
                     "emit":(3, total_size), "current":(total_size,),
                     "cs_invariant":(2, total_size)}
        self.monitor_init(group_name, save_every, buffer_size, total_size,
                          dict_buffer, dict_file, file_name, mpi_mode)
        
        self.dict_buffer = dict_buffer
        self.dict_file = dict_file
                    
    def track(self, object_to_save):
        """
        Save data
        
        Parameters
        ----------
        object_to_save : Bunch or Beam object
        """        
        self.track_bunch_data(object_to_save)
        
class PhaseSpaceMonitor(Monitor):
    """
    Monitor a single bunch and save the full phase space.
    
    Parameters
    ----------
    bunch_number : int
        Bunch to monitor
    mp_number : int or float
        Number of macroparticle in the phase space to save. If less than the 
        total number of macroparticles, a random fraction of the bunch is saved.
    save_every : int or float
        Set the frequency of the save. The data is saved every save_every 
        call of the montior.
    buffer_size : int or float
        Size of the save buffer.
    total_size : int or float
        Total size of the save. The following relationships between the 
        parameters must exist: 
            total_size % buffer_size == 0
            number of call to track / save_every == total_size
    file_name : string, optional
        Name of the HDF5 where the data will be stored. Must be specified
        the first time a subclass of Monitor is instancied and must be None
        the following times.
    mpi_mode : bool, optional
        If True, open the HDF5 file in parallel mode, which is needed to
        allow several cores to write in the same file at the same time.
        If False, open the HDF5 file in standard mode.

    Methods
    -------
    track(object_to_save)
        Save data
    """
    
    def __init__(self, bunch_number, mp_number, save_every, buffer_size, 
                 total_size, file_name=None, mpi_mode=False):
        
        self.bunch_number = bunch_number
        self.mp_number = int(mp_number)
        group_name = "PhaseSpaceData_" + str(self.bunch_number)
        dict_buffer = {"particles":(self.mp_number, 6, buffer_size), 
                       "alive":(self.mp_number, buffer_size)}
        dict_file = {"particles":(self.mp_number, 6, total_size),
                     "alive":(self.mp_number, total_size)}
        self.monitor_init(group_name, save_every, buffer_size, total_size,
                          dict_buffer, dict_file, file_name, mpi_mode)
        
        self.dict_buffer = dict_buffer
        self.dict_file = dict_file
                    
    def track(self, object_to_save):
        """
        Save data
        
        Parameters
        ----------
        object_to_save : Bunch or Beam object
        """        
        self.track_bunch_data(object_to_save)
        
    def to_buffer(self, bunch):
        """
        Save data to buffer.
        
        Parameters
        ----------
        bunch : Bunch object
        """
        self.time[self.buffer_count] = self.track_count
        
        if len(bunch.alive) != self.mp_number:
            index = np.arange(len(bunch.alive))
            samples_meta = random.sample(list(index), self.mp_number)
            samples = sorted(samples_meta)
        else:
            samples = slice(None)

        self.alive[:, self.buffer_count] = bunch.alive[samples]
        for i, dim in enumerate(bunch):
            self.particles[:, i, self.buffer_count] = bunch.particles[dim][samples]
        
        self.buffer_count += 1
        
        if self.buffer_count == self.buffer_size:
            self.write()
            self.buffer_count = 0
        

            
class BeamMonitor(Monitor):
    """
    Monitor the full beam and save each bunch attributes (mean, std, emit and 
    current).
    
    Parameters
    ----------
    h : int
        Harmonic number of the ring.
    save_every : int or float
        Set the frequency of the save. The data is saved every save_every 
        call of the montior.
    buffer_size : int or float
        Size of the save buffer.
    total_size : int or float
        Total size of the save. The following relationships between the 
        parameters must exist: 
            total_size % buffer_size == 0
            number of call to track / save_every == total_size
    file_name : string, optional
        Name of the HDF5 where the data will be stored. Must be specified
        the first time a subclass of Monitor is instancied and must be None
        the following times.
    mpi_mode : bool, optional
        If True, open the HDF5 file in parallel mode, which is needed to
        allow several cores to write in the same file at the same time.
        If False, open the HDF5 file in standard mode.

    Methods
    -------
    track(beam)
        Save data    
    """
    
    def __init__(self, h, save_every, buffer_size, total_size, file_name=None, 
                 mpi_mode=False):
        
        group_name = "Beam"
        dict_buffer = {"mean" : (6, h, buffer_size), 
                       "std" : (6, h, buffer_size),
                       "emit" : (3, h, buffer_size),
                       "current" : (h, buffer_size),
                       "cs_invariant" : (2, h, buffer_size)}
        dict_file = {"mean" : (6, h, total_size), 
                       "std" : (6, h, total_size),
                       "emit" : (3, h, total_size),
                       "current" : (h, total_size),
                       "cs_invariant" : (2, h, total_size)}
        
        self.monitor_init(group_name, save_every, buffer_size, total_size,
                          dict_buffer, dict_file, file_name, mpi_mode)
                    
    def track(self, beam):
        """
        Save data
        
        Parameters
        ----------
        beam : Beam object
        """     
        if self.track_count % self.save_every == 0:
            if (beam.mpi_switch == True):
                self.to_buffer(beam[beam.mpi.bunch_num], beam.mpi.bunch_num)
            else:
                self.to_buffer_no_mpi(beam)
                    
        self.track_count += 1
        
    def to_buffer(self, bunch, bunch_num):
        """
        Save data to buffer, if mpi is being used.
        
        Parameters
        ----------
        bunch : Bunch object
        bunch_num : int
        """
        
        self.time[self.buffer_count] = self.track_count
        self.mean[:, bunch_num, self.buffer_count] = bunch.mean
        self.std[:, bunch_num, self.buffer_count] = bunch.std
        self.emit[:, bunch_num, self.buffer_count] = bunch.emit
        self.current[bunch_num, self.buffer_count] = bunch.current
        self.cs_invariant[:, bunch_num, self.buffer_count] = bunch.cs_invariant
        
        self.buffer_count += 1
        
        if self.buffer_count == self.buffer_size:
            self.write(bunch_num)
            self.buffer_count = 0
            
    def to_buffer_no_mpi(self, beam):
        """
        Save data to buffer, if mpi is not being used.
        
        Parameters
        ----------
        beam : Beam object
        """
              
        self.time[self.buffer_count] = self.track_count
        self.mean[:, :, self.buffer_count] = beam.bunch_mean
        self.std[:, :, self.buffer_count] = beam.bunch_std
        self.emit[:, :, self.buffer_count] = beam.bunch_emit
        self.current[:, self.buffer_count] = beam.bunch_current
        self.cs_invariant[:, :, self.buffer_count] = beam.bunch_cs
        
        self.buffer_count += 1
        
        if self.buffer_count == self.buffer_size:
            self.write_no_mpi()
            self.buffer_count = 0

    def write(self, bunch_num):
        """
        Write data from buffer to the HDF5 file, if mpi is being used.
        
        Parameters
        ----------
        bunch_num : int
        """
        self.file[self.group_name]["time"][self.write_count*self.buffer_size:(
                    self.write_count+1)*self.buffer_size] = self.time
    
        self.file[self.group_name]["mean"][:, bunch_num, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.mean[:, bunch_num, :]
                 
        self.file[self.group_name]["std"][:, bunch_num, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.std[:, bunch_num, :]

        self.file[self.group_name]["emit"][:, bunch_num, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.emit[:, bunch_num, :]

        self.file[self.group_name]["current"][bunch_num, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.current[bunch_num, :]
        
        self.file[self.group_name]["cs_invariant"][:, bunch_num, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.cs_invariant[:, bunch_num, :]
                 
        self.file.flush() 
        self.write_count += 1
        
    def write_no_mpi(self):
        """
        Write data from buffer to the HDF5 file, if mpi is not being used.
        """
        
        self.file[self.group_name]["time"][self.write_count*self.buffer_size:(
                    self.write_count+1)*self.buffer_size] = self.time
    
        self.file[self.group_name]["mean"][:, :, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.mean
                 
        self.file[self.group_name]["std"][:, :, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.std

        self.file[self.group_name]["emit"][:, :, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.emit

        self.file[self.group_name]["current"][:, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.current
        
        self.file[self.group_name]["cs_invariant"][:, :, 
                 self.write_count*self.buffer_size:(self.write_count+1) * 
                 self.buffer_size] = self.cs_invariant
                 
        self.file.flush() 
        self.write_count += 1

        
class ProfileMonitor(Monitor):
    """
    Monitor a single bunch and save bunch profiles.
    
    Parameters
    ----------
    bunch_number : int
        Bunch to monitor.
    save_every : int or float
        Set the frequency of the save. The data is saved every save_every 
        call of the montior.
    buffer_size : int or float
        Size of the save buffer.
    total_size : int or float
        Total size of the save. The following relationships between the 
        parameters must exist: 
            total_size % buffer_size == 0
            number of call to track / save_every == total_size
    dimensions : str or list of str, optional
        Dimensions to save.
    n_bin : int or list of int, optional
        Number of bin to use in each dimension.
    file_name : string, optional
        Name of the HDF5 where the data will be stored. Must be specified
        the first time a subclass of Monitor is instancied and must be None
        the following times.
    mpi_mode : bool, optional
        If True, open the HDF5 file in parallel mode, which is needed to
        allow several cores to write in the same file at the same time.
        If False, open the HDF5 file in standard mode.

    Methods
    -------
    track(object_to_save)
        Save data.
    """
    
    def __init__(self, bunch_number, save_every, buffer_size, total_size, 
                 dimensions="tau", n_bin=75, file_name=None, mpi_mode=False):
        
        self.bunch_number = bunch_number
        group_name = "ProfileData_" + str(self.bunch_number)
        
        if isinstance(dimensions, str):
            self.dimensions = [dimensions]
        else:
            self.dimensions = dimensions
            
        if isinstance(n_bin, int):
            self.n_bin = np.ones((len(self.dimensions),), dtype=int)*n_bin
        else:
            self.n_bin = n_bin
        
        dict_buffer = {}
        dict_file = {}
        for index, dim in enumerate(self.dimensions):
            dict_buffer.update({dim : (self.n_bin[index] - 1, buffer_size)})
            dict_buffer.update({dim + "_bin" : (self.n_bin[index] - 1, buffer_size)})
            dict_file.update({dim : (self.n_bin[index] - 1, total_size)})
            dict_file.update({dim + "_bin" : (self.n_bin[index] - 1, total_size)})

        self.monitor_init(group_name, save_every, buffer_size, total_size,
                          dict_buffer, dict_file, file_name, mpi_mode)
        
        self.dict_buffer = dict_buffer
        self.dict_file = dict_file
        
    def to_buffer(self, bunch):
        """
        Save data to buffer.
        
        Parameters
        ----------
        bunch : Bunch object
        """

        self.time[self.buffer_count] = self.track_count
        for index, dim in enumerate(self.dimensions):
            bins, sorted_index, profile, center = bunch.binning(dim, self.n_bin[index])
            self.__getattribute__(dim + "_bin")[:, self.buffer_count] = center
            self.__getattribute__(dim)[:, self.buffer_count] = profile
        
        self.buffer_count += 1
        
        if self.buffer_count == self.buffer_size:
            self.write()
            self.buffer_count = 0
            
    def write(self):
        """Write data from buffer to the HDF5 file."""
        
        self.file[self.group_name]["time"][self.write_count*self.buffer_size:(
                    self.write_count+1)*self.buffer_size] = self.time

        for dim in self.dimensions:
            self.file[self.group_name][dim][:, 
                    self.write_count * self.buffer_size:(self.write_count+1) * 
                    self.buffer_size] = self.__getattribute__(dim)
            self.file[self.group_name][dim + "_bin"][:, 
                    self.write_count * self.buffer_size:(self.write_count+1) * 
                    self.buffer_size] = self.__getattribute__(dim + "_bin")
            
        self.write_count += 1
                    
    def track(self, object_to_save):
        """
        Save data.
        
        Parameters
        ----------
        object_to_save : Bunch or Beam object
        """        
        self.track_bunch_data(object_to_save, check_empty=True)
        
class WakePotentialMonitor(Monitor):
    """
    Monitor the wake potential from a single bunch and save attributes (tau, 
    ...).
    
    Parameters
    ----------
    bunch_number : int
        Bunch to monitor.
    wake_types : str or list of str
        Wake types to save: "Wlong, "Wxdip", ...
    n_bin : int
        Number of bin to be used to interpolate the wake potential on a fixed
        grid.
    save_every : int or float
        Set the frequency of the save. The data is saved every save_every 
        call of the montior.
    buffer_size : int or float
        Size of the save buffer.
    total_size : int or float
        Total size of the save. The following relationships between the 
        parameters must exist: 
            total_size % buffer_size == 0
            number of call to track / save_every == total_size
    file_name : string, optional
        Name of the HDF5 where the data will be stored. Must be specified
        the first time a subclass of Monitor is instancied and must be None
        the following times.
    mpi_mode : bool, optional
        If True, open the HDF5 file in parallel mode, which is needed to
        allow several cores to write in the same file at the same time.
        If False, open the HDF5 file in standard mode.

    Methods
    -------
    track(object_to_save, wake_potential_to_save)
        Save data.
    """
    
    def __init__(self, bunch_number, wake_types, n_bin, save_every, 
                 buffer_size, total_size, file_name=None, mpi_mode=False):
        
        self.bunch_number = bunch_number
        group_name = "WakePotentialData_" + str(self.bunch_number)
        
        if isinstance(wake_types, str):
            self.wake_types = [wake_types]
        else:
            self.wake_types = wake_types
            
        self.n_bin = n_bin*2
        
        dict_buffer = {}
        dict_file = {}
        for index, dim in enumerate(self.wake_types):
            dict_buffer.update({"tau_" + dim : (self.n_bin, buffer_size)})
            dict_file.update({"tau_" + dim : (self.n_bin, total_size)})
            dict_buffer.update({"profile_" + dim : (self.n_bin, buffer_size)})
            dict_file.update({"profile_" + dim : (self.n_bin, total_size)})
            dict_buffer.update({dim : (self.n_bin, buffer_size)})
            dict_file.update({dim : (self.n_bin, total_size)})
            if dim == "Wxdip" or dim == "Wydip":
                dict_buffer.update({"dipole_" + dim : (self.n_bin, buffer_size)})
                dict_file.update({"dipole_" + dim : (self.n_bin, total_size)})

        self.monitor_init(group_name, save_every, buffer_size, total_size,
                          dict_buffer, dict_file, file_name, mpi_mode)
        
        self.dict_buffer = dict_buffer
        self.dict_file = dict_file
        
    def to_buffer(self, wp):
        """
        Save data to buffer.
        
        Parameters
        ----------
        wp : WakePotential object
        """

        self.time[self.buffer_count] = self.track_count
        for index, dim in enumerate(self.wake_types):
            tau0 = wp.__getattribute__("tau0_" + dim)
            profile0 = wp.__getattribute__("profile0_" + dim)
            WP0 = wp.__getattribute__(dim)
            if dim == "Wxdip":
                dipole0 = wp.__getattribute__("dipole_x")
            if dim == "Wydip":
                dipole0 = wp.__getattribute__("dipole_y")
            
            tau = np.linspace(tau0[0], tau0[-1], self.n_bin)
            f = interp1d(tau0, WP0, fill_value = 0, bounds_error = False)
            WP = f(tau)
            g = interp1d(tau0, profile0, fill_value = 0, bounds_error = False)
            profile = g(tau)
            if dim == "Wxdip" or dim == "Wydip":
                h = interp1d(tau0, dipole0, fill_value = 0, bounds_error = False)
                dipole = h(tau)
            
            self.__getattribute__("tau_" + dim)[:, self.buffer_count] = tau + wp.tau_mean
            self.__getattribute__("profile_" + dim)[:, self.buffer_count] = profile
            self.__getattribute__(dim)[:, self.buffer_count] = WP
            if dim == "Wxdip" or dim == "Wydip":
                self.__getattribute__("dipole_" + dim)[:, self.buffer_count] = dipole
            
        self.buffer_count += 1
        
        if self.buffer_count == self.buffer_size:
            self.write()
            self.buffer_count = 0
            
    def write(self):
        """Write data from buffer to the HDF5 file."""
        
        self.file[self.group_name]["time"][self.write_count*self.buffer_size:(
                    self.write_count+1)*self.buffer_size] = self.time
        
        for dim in self.wake_types:
            self.file[self.group_name]["tau_" + dim][:, 
                    self.write_count * self.buffer_size:(self.write_count+1) * 
                    self.buffer_size] = self.__getattribute__("tau_" + dim)
            self.file[self.group_name]["profile_" + dim][:, 
                    self.write_count * self.buffer_size:(self.write_count+1) * 
                    self.buffer_size] = self.__getattribute__("profile_" + dim)
            self.file[self.group_name][dim][:, 
                    self.write_count * self.buffer_size:(self.write_count+1) * 
                    self.buffer_size] = self.__getattribute__(dim)
            if dim == "Wxdip" or dim == "Wydip":
                self.file[self.group_name]["dipole_" + dim][:, 
                    self.write_count * self.buffer_size:(self.write_count+1) * 
                    self.buffer_size] = self.__getattribute__("dipole_" + dim)
            
        self.file.flush()
        self.write_count += 1
                    
    def track(self, object_to_save, wake_potential_to_save):
        """
        Save data.
        
        Parameters
        ----------
        object_to_save : Bunch or Beam object
        wake_potential_to_save : WakePotential object
        """
        if isinstance(object_to_save, Beam):
            if (object_to_save.mpi_switch == True):
                if object_to_save.mpi.bunch_num == self.bunch_number:
                    save = True
                else:
                    save = False
            else:
                raise NotImplementedError("WakePotentialMonitor for Beam " +
                                          "objects is only available " +
                                          "with MPI mode.")
        elif isinstance(object_to_save, Bunch):
            save = True
        else:                            
            raise TypeError("object_to_save should be a Beam or Bunch object.")
            
        if save and (self.track_count % self.save_every == 0):
            self.to_buffer(wake_potential_to_save)
        self.track_count += 1
    
class BunchSpectrumMonitor(Monitor):
    """
    Monitor the coherent and incoherent bunch spectrums. 
    
    Parameters
    ----------
    ring : Synchrotron object
    bunch_number : int
        Bunch to monitor
    mp_number : int or float
        Total number of macro-particles in the bunch.
    sample_size : int or float
        Number of macro-particles to be used for tune and FFT computation.
        This number cannot exceed mp_number.  
    save_every : int or float
        Set the frequency of the save. The spectrums are computed every 
        save_every call of the montior.
    buffer_size : int or float
        Size of the save buffer.
    total_size : int or float
        Total size of the save. The following relationships between the 
        parameters must exist: 
            total_size % buffer_size == 0
            number of call to track / save_every == total_size - 1 
    dim : str, optional
        Dimensions in which the spectrums have to be computed.
        Can be:
                - "all"
                - "tau"
                - "x"
                - "y"
                - "xy" or "yx"
                - "xtau" or "taux"
                - "ytau" or "tauy"
    n_fft : int or float, optional
        The number of points used for FFT computation, if n_fft is bigger than
        save_every zero-padding is applied.
        If None, save_every is used.
    file_name : string, optional
        Name of the HDF5 where the data will be stored. Must be specified
        the first time a subclass of Monitor is instancied and must be None
        the following times.
    mpi_mode : bool, optional
        If True, open the HDF5 file in parallel mode, which is needed to
        allow several cores to write in the same file at the same time.
        If False, open the HDF5 file in standard mode.
        
    Attributes
    ----------
    fft_resolution : float
        Return the fft resolution in [Hz].
    signal_resolution : float
        Return the signal resolution in [Hz].
    frequency_samples : array of float
        Return the fft frequency samples in [Hz].        
        
    Methods
    -------
    track(bunch):
        Save spectrum data.
    
    """
    
    def __init__(self, ring, bunch_number, mp_number, sample_size, save_every, 
                 buffer_size, total_size, dim="all", n_fft=None, 
                 file_name=None, mpi_mode=False):
        
        if n_fft is None:
            self.n_fft = int(save_every)
        else:
            self.n_fft = int(n_fft)
            
        self.sample_size = int(sample_size)
        self.store_dict = {"x":0,"y":1,"tau":2}

        if dim == "all":
            self.track_dict = {"x":0,"y":1,"tau":2}
            self.mean_index = [True, False, True, False, True, False]
        elif dim == "tau":
            self.track_dict = {"tau":0}
            self.mean_index = [False, False, False, False, True, False]
        elif dim == "x":
            self.track_dict = {"x":0}
            self.mean_index = [True, False, False, False, False, False]
        elif dim == "y":
            self.track_dict = {"y":0}
            self.mean_index = [False, False, True, False, False, False]
        elif dim == "xy" or dim == "yx":
            self.track_dict = {"x":0,"y":1}
            self.mean_index = [True, False, True, False, False, False]
        elif dim == "xtau" or dim == "taux":
            self.track_dict = {"x":0,"tau":1}
            self.mean_index = [True, False, False, False, True, False]
        elif dim == "ytau" or dim == "tauy":
            self.track_dict = {"y":0,"tau":1}
            self.mean_index = [False, False, True, False, True, False]
        else:
            raise ValueError("dim is not correct.")
        
        self.size_list = len(self.track_dict)
        
        self.ring = ring
        self.bunch_number = bunch_number
        group_name = "BunchSpectrum_" + str(self.bunch_number)
        
        dict_buffer = {"incoherent":(3, self.n_fft//2+1, buffer_size),
                       "coherent":(3, self.n_fft//2+1, buffer_size),
                       "mean_incoherent":(3,buffer_size),
                       "std_incoherent":(3,buffer_size)}
        dict_file = {"incoherent":(3, self.n_fft//2+1, total_size),
                        "coherent":(3, self.n_fft//2+1, total_size),
                        "mean_incoherent":(3,total_size),
                        "std_incoherent":(3,total_size)}
        
        self.monitor_init(group_name, save_every, buffer_size, total_size,
                          dict_buffer, dict_file, file_name, mpi_mode)
        
        self.dict_buffer = dict_buffer
        self.dict_file = dict_file
        
        self.save_count = 0
        
        self.positions = np.zeros((self.size_list, self.sample_size, self.save_every+1))
        self.mean = np.zeros((self.size_list, self.save_every+1))
        
        index = np.arange(0, int(mp_number))
        self.index_sample = sorted(random.sample(list(index), self.sample_size))
                
        self.incoherent = np.zeros((3, self.n_fft//2+1, self.buffer_size))
        self.coherent = np.zeros((3, self.n_fft//2+1, self.buffer_size))
        
        self.file[self.group_name].create_dataset(
            "freq", data=self.frequency_samples)

    @property
    def fft_resolution(self):
        """
        Return the fft resolution in [Hz].
        
        It is defined as the sampling frequency over the number of samples.
        """
        return self.ring.f0/self.n_fft
    
    @property
    def signal_resolution(self):
        """
        Return the signal resolution in [Hz].
        
        It is defined as the inverse of the signal length.
        """
        return 1/(self.ring.T0*self.save_every)
    
    @property
    def frequency_samples(self):
        """
        Return the fft frequency samples in [Hz].
        """
        return rfftfreq(self.n_fft, self.ring.T0)    
        
    def track(self, object_to_save):
        """
        Save spectrum data.

        Parameters
        ----------
        object_to_save : Beam or Bunch object

        """
        save = True
        if isinstance(object_to_save, Beam):
            if (object_to_save.mpi_switch == True):
                if object_to_save.mpi.bunch_num == self.bunch_number:
                    bunch = object_to_save[object_to_save.mpi.bunch_num]
                else:
                    save = False
            else:
                bunch = object_to_save[self.bunch_number]
        elif isinstance(object_to_save, Bunch):
            bunch = object_to_save
        else:
            raise TypeError("object_to_save should be a Beam or Bunch object.")
        
        if save:
            try:
                for key, value in self.track_dict.items():
                    self.positions[value, :, self.save_count] = bunch[key][self.index_sample]
            except IndexError:
                self.positions[value, :, self.save_count] = np.nan
            
            self.mean[:, self.save_count] = bunch.mean[self.mean_index]
            
            self.save_count += 1
            
            if self.track_count > 0 and self.track_count % self.save_every == 0:
                self.to_buffer(bunch)
                self.save_count = 0
    
            self.track_count += 1
        
    def to_buffer(self, bunch):
        """
        A method to hold saved data before writing it to the output file.

        """
        
        self.time[self.buffer_count] = self.track_count
        
        for key, value in self.track_dict.items():
            incoherent, mean_incoherent, std_incoherent = self.get_incoherent_spectrum(self.positions[value,:,:])
            self.incoherent[self.store_dict[key],:,self.buffer_count] = incoherent
            self.mean_incoherent[self.store_dict[key],self.buffer_count] = mean_incoherent
            self.std_incoherent[self.store_dict[key],self.buffer_count] = std_incoherent
            self.coherent[self.store_dict[key],:,self.buffer_count] = self.get_coherent_spectrum(self.mean[value])
        
        self.buffer_count += 1
        
        if self.buffer_count == self.buffer_size:
            self.write()
            self.buffer_count = 0
            
    def write(self):
        """
        Write data from buffer to output file.

        """
        self.file[self.group_name]["time"][self.write_count*self.buffer_size:(
                    self.write_count+1)*self.buffer_size] = self.time

        self.file[self.group_name]["incoherent"][:,:, 
                self.write_count * self.buffer_size:(self.write_count+1) * 
                self.buffer_size] = self.incoherent
        self.file[self.group_name]["mean_incoherent"][:, 
                self.write_count * self.buffer_size:(self.write_count+1) * 
                self.buffer_size] = self.mean_incoherent
        self.file[self.group_name]["std_incoherent"][:, 
                self.write_count * self.buffer_size:(self.write_count+1) * 
                self.buffer_size] = self.std_incoherent
        self.file[self.group_name]["coherent"][:,:, 
                self.write_count * self.buffer_size:(self.write_count+1) * 
                self.buffer_size] = self.coherent
            
        self.file.flush()
        self.write_count += 1
    
    def get_incoherent_spectrum(self, positions):
        """
        Compute the incoherent spectrum i.e. the average of the absolute value 
        of the FT of the position of every particule of the bunch. 

        Returns
        -------
        incoherent : array
            Bunch incoherent spectrum.
        mean_incoherent : float
            Mean frequency of the maximum of each individual particle spectrum
            in [Hz].
        std_incoherent : float
            Standard deviation of the frequency of the maximum of each 
            individual particle spectrum in [Hz].

        """
        fourier = rfft(positions, n=self.n_fft)
        fourier_abs = np.abs(fourier)
        max_array = np.argmax(fourier_abs,axis=1)
        freq_array = self.frequency_samples[max_array]
        mean_incoherent = np.mean(freq_array)
        std_incoherent = np.std(freq_array)
        incoherent = np.mean(fourier_abs, axis=0)
        
        return incoherent, mean_incoherent, std_incoherent
    
    def get_coherent_spectrum(self, mean):
        """
        Compute the coherent spectrum i.e. the absolute value of the FT of the
        mean position of the bunch.

        Returns
        -------
        coherent : array
            Bunch coherent spectrum.

        """
        coherent = np.abs(rfft(mean, n=self.n_fft))
        
        return coherent
    
class BeamSpectrumMonitor(Monitor):
    """
    Monitor coherent beam spectrum. 
    
    Parameters
    ----------
    ring : Synchrotron object
    save_every : int or float
        Set the frequency of the save. The spectrums are computed every 
        save_every call of the montior.
    buffer_size : int or float
        Size of the save buffer.
    total_size : int or float
        Total size of the save. The following relationships between the 
        parameters must exist: 
            total_size % buffer_size == 0
            number of call to track / save_every == total_size - 1 
    dim : str, optional
        Dimensions in which the spectrums have to be computed.
        Can be:
                - "all"
                - "tau"
                - "x"
                - "y"
                - "xy" or "yx"
                - "xtau" or "taux"
                - "ytau" or "tauy"
    n_fft : int or float, optional
        The number of points used for FFT computation, if n_fft is bigger than
        save_every zero-padding is applied.
        If None, save_every is used.
    file_name : string, optional
        Name of the HDF5 where the data will be stored. Must be specified
        the first time a subclass of Monitor is instancied and must be None
        the following times.
    mpi_mode : bool, optional
        If True, open the HDF5 file in parallel mode, which is needed to
        allow several cores to write in the same file at the same time.
        If False, open the HDF5 file in standard mode.
        
    Attributes
    ----------
    fft_resolution : float
        Return the fft resolution in [Hz].
    signal_resolution : float
        Return the signal resolution in [Hz].
    frequency_samples : array of float
        Return the fft frequency samples in [Hz].        
        
    Methods
    -------
    track(bunch):
        Save spectrum data.
    
    """
    
    def __init__(self, ring, save_every, buffer_size, total_size, dim="all", 
                 n_fft=None, file_name=None, mpi_mode=False):
        
        if n_fft is None:
            self.n_fft = int(save_every)
        else:
            self.n_fft = int(n_fft)
            
        self.store_dict = {"x":0,"y":1,"tau":2}

        if dim == "all":
            self.track_dict = {"x":0,"y":1,"tau":2}
            self.mean_index = [True, False, True, False, True, False]
        elif dim == "tau":
            self.track_dict = {"tau":0}
            self.mean_index = [False, False, False, False, True, False]
        elif dim == "x":
            self.track_dict = {"x":0}
            self.mean_index = [True, False, False, False, False, False]
        elif dim == "y":
            self.track_dict = {"y":0}
            self.mean_index = [False, False, True, False, False, False]
        elif dim == "xy" or dim == "yx":
            self.track_dict = {"x":0,"y":1}
            self.mean_index = [True, False, True, False, False, False]
        elif dim == "xtau" or dim == "taux":
            self.track_dict = {"x":0,"tau":1}
            self.mean_index = [True, False, False, False, True, False]
        elif dim == "ytau" or dim == "tauy":
            self.track_dict = {"y":0,"tau":1}
            self.mean_index = [False, False, True, False, True, False]
        else:
            raise ValueError("dim is not correct.")
        
        self.size_list = len(self.track_dict)
        
        self.ring = ring
        group_name = "BeamSpectrum"
        
        dict_buffer = {"coherent":(3, self.n_fft//2+1, buffer_size)}
        dict_file = {"coherent":(3, self.n_fft//2+1, total_size)}
        
        self.monitor_init(group_name, save_every, buffer_size, total_size,
                          dict_buffer, dict_file, file_name, mpi_mode)
        
        self.dict_buffer = dict_buffer
        self.dict_file = dict_file
        
        self.save_count = 0
        
        self.mean = np.zeros((self.size_list, ring.h, self.save_every))
        self.coherent = np.zeros((3, self.n_fft//2+1, self.buffer_size))
        
        self.file[self.group_name].create_dataset(
            "freq", data=self.frequency_samples)
        
    @property
    def fft_resolution(self):
        """
        Return the fft resolution in [Hz].
        
        It is defined as the sampling frequency over the number of samples.
        """
        return self.ring.f1/self.n_fft
    
    @property
    def signal_resolution(self):
        """
        Return the signal resolution in [Hz].
        
        It is defined as the inverse of the signal length.
        """
        return 1/(self.ring.T0*self.save_every)
    
    @property
    def frequency_samples(self):
        """
        Return the fft frequency samples in [Hz].
        """
        return rfftfreq(self.n_fft, self.ring.T1)
        
    def track(self, beam):
        """
        Save mean data.

        Parameters
        ----------
        beam : Beam object

        """
        if (beam.mpi_switch == True):
            bunch_num = beam.mpi.bunch_num
            bunch = beam[bunch_num]
            self.mean[:, bunch_num, self.save_count] = bunch.mean[self.mean_index]
        else:
            self.mean[:, :, self.save_count] = beam.bunch_mean[self.mean_index,:]
            
        self.save_count += 1
            
        if self.save_count == self.save_every:
            self.to_buffer(beam)
            self.save_count = 0
        
        self.track_count += 1

    def to_buffer(self, beam):
        """
        A method to hold saved data before writing it to the output file.

        """
        
        self.time[self.buffer_count] = self.track_count
        
        for key, value in self.track_dict.items():
            if (beam.mpi_switch == True):
                data_core = self.mean[value, beam.mpi.bunch_num, :]
                full_data = beam.mpi.comm.allgather(data_core)
                data = np.reshape(full_data, (-1), 'F')
            else:
                data = np.reshape(self.mean[value, :, :], (-1), 'F')
            self.coherent[self.store_dict[key],:,self.buffer_count] = self.get_beam_spectrum(data)        
        self.buffer_count += 1
        
        if self.buffer_count == self.buffer_size:
            self.write()
            self.buffer_count = 0
            
    def write(self):
        """
        Write data from buffer to output file.

        """
        self.file[self.group_name]["time"][self.write_count*self.buffer_size:(
                    self.write_count+1)*self.buffer_size] = self.time

        self.file[self.group_name]["coherent"][:,:, 
                self.write_count * self.buffer_size:(self.write_count+1) * 
                self.buffer_size] = self.coherent
            
        self.file.flush()
        self.write_count += 1
    
    def get_beam_spectrum(self, mean):
        """
        Compute the beam coherent spectrum i.e. the absolute value of the FT 
        of the mean position of every bunch.

        Returns
        -------
        coherent : array
            The beam coherent spectrum.

        """
        coherent = np.abs(rfft(mean, n=self.n_fft))
        
        return coherent
        
class CavityMonitor(Monitor):
    """
    Monitor a CavityResonator object and save attributes.
    
    Parameters
    ----------
    cavity_name : str
        Name of the CavityResonator object to monitor.
    ring : Synchrotron object
    save_every : int or float
        Set the frequency of the save. The data is saved every save_every 
        call of the montior.
    buffer_size : int or float
        Size of the save buffer.
    total_size : int or float
        Total size of the save. The following relationships between the 
        parameters must exist: 
            total_size % buffer_size == 0
            number of call to track / save_every == total_size
    file_name : string, optional
        Name of the HDF5 where the data will be stored. Must be specified
        the first time a subclass of Monitor is instancied and must be None
        the following times.
    mpi_mode : bool, optional
        If True, open the HDF5 file in parallel mode, which is needed to
        allow several cores to write in the same file at the same time.
        If False, open the HDF5 file in standard mode.

    Methods
    -------
    track(beam, cavity)
        Save data
    """
    
    def __init__(self, cavity_name, ring, save_every, buffer_size, total_size, 
                 file_name=None, mpi_mode=False):
        
        self.cavity_name = cavity_name
        self.ring = ring
        
        group_name = cavity_name
        dict_buffer = {"cavity_phasor_record":(ring.h, buffer_size,),
                       "beam_phasor_record":(ring.h, buffer_size,),
                       "detune":(buffer_size,),
                       "psi":(buffer_size,),
                       "Vg":(buffer_size,),
                       "theta_g":(buffer_size,),
                       "Pg":(buffer_size,),
                       "Rs":(buffer_size,),
                       "Q":(buffer_size,),
                       "QL":(buffer_size,)}
        dict_file = {"cavity_phasor_record":(ring.h, total_size,),
                     "beam_phasor_record":(ring.h, total_size,),
                     "detune":(total_size,),
                     "psi":(total_size,),
                     "Vg":(total_size,),
                     "theta_g":(total_size,),
                     "Pg":(total_size,),
                     "Rs":(total_size,),
                     "Q":(total_size,),
                     "QL":(total_size,)}
        dict_dtype = {"cavity_phasor_record":complex,
                      "beam_phasor_record":complex,
                      "detune":float,
                      "psi":float,
                      "Vg":float,
                      "theta_g":float,
                      "Pg":float,
                      "Rs":float,
                      "Q":float,
                      "QL":float}
        
        self.monitor_init(group_name, save_every, buffer_size, total_size,
                          dict_buffer, dict_file, file_name, mpi_mode, 
                          dict_dtype)
        
        self.dict_buffer = dict_buffer
        self.dict_file = dict_file
                    
    def track(self, beam, cavity):
        """
        Save data
        
        Parameters
        ----------
        beam : Beam object
        cavity : CavityResonator object
        """        
        if self.track_count % self.save_every == 0:
            if isinstance(cavity, CavityResonator):
                if beam.mpi_switch == False:
                    self.to_buffer(cavity)
                elif beam.mpi.rank == 0:
                    self.to_buffer(cavity)
                else:
                    pass
            else:                            
                raise TypeError("cavity should be a CavityResonator object.")
        self.track_count += 1       