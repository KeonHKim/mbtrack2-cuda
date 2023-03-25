# -*- coding: utf-8 -*-
"""
Module where particles, bunches and beams are described as objects.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.constants import c, m_e, m_p, e

class Particle:
    """
    Define a particle object.

    Attributes
    ----------
    mass : float
        total particle mass in [kg]
    charge : float
        electrical charge in [C]
    E_rest : float
        particle rest energy in [eV]
    """
    def __init__(self, mass, charge):
        self.mass = mass
        self.charge = charge

    @property
    def E_rest(self):
        return self.mass * c ** 2 / e
    
class Electron(Particle):
    """ Define an electron"""
    def __init__(self):
        super().__init__(m_e, -1*e)
        
class Proton(Particle):
    """ Define a proton"""
    def __init__(self):
        super().__init__(m_p, e)

class Bunch:
    """
    Define a bunch object.
    
    Parameters
    ----------
    ring : Synchrotron object
    mp_number : float, optional
        Macro-particle number.
    current : float, optional
        Bunch current in [A].
    track_alive : bool, optional
        If False, the code no longer take into account alive/dead particles.
        Should be set to True if element such as apertures are used.
        Can be set to False to gain a speed increase.
    alive : bool, optional
        If False, the bunch is defined as empty.
        
    Attributes
    ----------
    mp_number : int
        Macro-particle number.
    alive : array of bool of shape (mp_number,)
        Array used to monitor which particle is dead or alive.
    charge : float
        Bunch charge in [C].
    charge_per_mp : float
        Charge per macro-particle in [C].
    particle_number : int
        Number of particles in the bunch.
    current : float
        Bunch current in [A].
    is_empty : bool
        Return True if the bunch is empty.
    mean : array of shape (6,)
        Mean position of alive particles for each coordinates.
    std : array of shape (6,)
        Standard deviation of the position of alive particles for each 
        coordinates.
    emit : array of shape (3,)
        Bunch emittance for each plane [1]. !!! -> Correct for long ?
        
    Methods
    -------
    init_gaussian(cov=None, mean=None, **kwargs)
        Initialize bunch particles with 6D gaussian phase space.
    plot_phasespace(x_var="tau", y_var="delta", plot_type="j")
        Plot phase space.
        
    References
    ----------
    [1] Wiedemann, H. (2015). Particle accelerator physics. 4th edition. 
    Springer, Eq.(8.39) of p224.
    """
    
    def __init__(self, ring, mp_number=1e3, current=1e-3, track_alive=True,
                 alive=True):
        
        self.ring = ring
        if not alive:
            mp_number = 1
            current = 0
        self._mp_number = int(mp_number)
        
        self.dtype = np.dtype([('x',np.float),
                       ('xp',np.float),
                       ('y',np.float),
                       ('yp',np.float),
                       ('tau',np.float),
                       ('delta',np.float)])
        
        self.particles = np.zeros(self.mp_number, self.dtype)
        self.track_alive = track_alive
        
        self.alive = np.ones((self.mp_number,),dtype=bool)
        self.current = current
        if not alive:
            self.alive = np.zeros((self.mp_number,),dtype=bool)
        
    def __len__(self):
        """Return the number of alive particles"""
        return len(self[:])
        
    def __getitem__(self, label):
        """Return the columns label for alive particles"""
        if self.track_alive is True:
            return self.particles[label][self.alive]
        else:
            return self.particles[label]
    
    def __setitem__(self, label, value):
        """Set value to the columns label for alive particles"""
        if self.track_alive is True:
            self.particles[label][self.alive] = value
        else:
            self.particles[label] = value
    
    def __iter__(self):
        """Iterate over labels"""
        return self.dtype.names.__iter__()
    
    def __repr__(self):
        """Return representation of alive particles"""
        return f'{pd.DataFrame(self[:])!r}'
        
    @property
    def mp_number(self):
        """Macro-particle number"""
        return self._mp_number
    
    @mp_number.setter
    def mp_number(self, value):
        self._mp_number = int(value)
        self.__init__(self.ring, value, self.charge)
        
    @property
    def charge_per_mp(self):
        """Charge per macro-particle [C]"""
        return self._charge_per_mp
    
    @charge_per_mp.setter
    def charge_per_mp(self, value):
        self._charge_per_mp = value
        
    @property
    def charge(self):
        """Bunch charge in [C]"""
        return self.__len__()*self.charge_per_mp
    
    @charge.setter
    def charge(self, value):
        self.charge_per_mp = value / self.__len__()
    
    @property
    def particle_number(self):
        """Particle number"""
        return int(self.charge / np.abs(self.ring.particle.charge))
    
    @particle_number.setter
    def particle_number(self, value):
        self.charge_per_mp = value * self.ring.particle.charge / self.__len__()
        
    @property
    def current(self):
        """Bunch current [A]"""
        return self.charge / self.ring.T0
    
    @current.setter
    def current(self, value):
        self.charge_per_mp = value * self.ring.T0 / self.__len__()
        
    @property
    def is_empty(self):
        """Return True if the bunch is empty."""
        return ~np.any(self.alive)
    
    @property    
    def mean(self):
        """
        Return the mean position of alive particles for each coordinates.
        """
        mean = [[self[name].mean()] for name in self]
        return np.squeeze(np.array(mean))
    
    @property
    def std(self):
        """
        Return the standard deviation of the position of alive 
        particles for each coordinates.
        """
        std = [[self[name].std()] for name in self]
        return np.squeeze(np.array(std))
    
    @property    
    def emit(self):
        """
        Return the bunch emittance for each plane.
        """
        emitX = (np.mean(self['x']**2)*np.mean(self['xp']**2) - 
                 np.mean(self['x']*self['xp'])**2)**(0.5)
        emitY = (np.mean(self['y']**2)*np.mean(self['yp']**2) - 
                 np.mean(self['y']*self['yp'])**2)**(0.5)
        emitS = (np.mean(self['tau']**2)*np.mean(self['delta']**2) - 
                 np.mean(self['tau']*self['delta'])**2)**(0.5)
        return np.array([emitX, emitY, emitS])
    
    @property
    def cs_invariant(self):
        """
        Return the average Courant-Snyder invariant of each plane.

        """
        Jx = (self.ring.optics.local_gamma[0] * self['x']**2) + \
              (2*self.ring.optics.local_alpha[0] * self['x'])*self['xp'] + \
              (self.ring.optics.local_beta[0] * self['xp']**2)
        Jy = (self.ring.optics.local_gamma[1] * self['y']**2) + \
              (2*self.ring.optics.local_alpha[1] * self['y']*self['yp']) + \
              (self.ring.optics.local_beta[1] * self['yp']**2)
        return np.array((np.mean(Jx),np.mean(Jy)))
        
    def init_gaussian(self, cov=None, mean=None, **kwargs):
        """
        Initialize bunch particles with 6D gaussian phase space.
        Covariance matrix is taken from [1] and dispersion is added following
        the method explained in [2].
                
        Parameters
        ----------
        cov : (6,6) array, optional
            Covariance matrix of the bunch distribution
        mean : (6,) array, optional
            Mean of the bunch distribution
        
        References
        ----------
        [1] Wiedemann, H. (2015). Particle accelerator physics. 4th 
        edition. Springer, Eq.(8.38) of p223.
        [2] http://www.pp.rhul.ac.uk/bdsim/manual-develop/dev_beamgeneration.html

        """
        if mean is None:
            mean = np.zeros((6,))
        
        if cov is None:
            sigma_0 = kwargs.get("sigma_0", self.ring.sigma_0)
            sigma_delta = kwargs.get("sigma_delta", self.ring.sigma_delta)
            optics = kwargs.get("optics", self.ring.optics)
            
            cov = np.zeros((6,6))
            cov[0,0] = self.ring.emit[0]*optics.local_beta[0] + (optics.local_dispersion[0]*self.ring.sigma_delta)**2
            cov[1,1] = self.ring.emit[0]*optics.local_gamma[0] + (optics.local_dispersion[1]*self.ring.sigma_delta)**2
            cov[0,1] = -1*self.ring.emit[0]*optics.local_alpha[0] + (optics.local_dispersion[0]*optics.local_dispersion[1]*self.ring.sigma_delta**2)
            cov[1,0] = -1*self.ring.emit[0]*optics.local_alpha[0] + (optics.local_dispersion[0]*optics.local_dispersion[1]*self.ring.sigma_delta**2)
            cov[0,5] = optics.local_dispersion[0]*self.ring.sigma_delta**2
            cov[5,0] = optics.local_dispersion[0]*self.ring.sigma_delta**2
            cov[1,5] = optics.local_dispersion[1]*self.ring.sigma_delta**2
            cov[5,1] = optics.local_dispersion[1]*self.ring.sigma_delta**2
            cov[2,2] = self.ring.emit[1]*optics.local_beta[1] + (optics.local_dispersion[2]*self.ring.sigma_delta)**2
            cov[3,3] = self.ring.emit[1]*optics.local_gamma[1] + (optics.local_dispersion[3]*self.ring.sigma_delta)**2
            cov[2,3] = -1*self.ring.emit[1]*optics.local_alpha[1] + (optics.local_dispersion[2]*optics.local_dispersion[3]*self.ring.sigma_delta**2)
            cov[3,2] = -1*self.ring.emit[1]*optics.local_alpha[1] + (optics.local_dispersion[2]*optics.local_dispersion[3]*self.ring.sigma_delta**2)
            cov[2,5] = optics.local_dispersion[2]*self.ring.sigma_delta**2
            cov[5,2] = optics.local_dispersion[2]*self.ring.sigma_delta**2
            cov[3,5] = optics.local_dispersion[3]*self.ring.sigma_delta**2
            cov[5,3] = optics.local_dispersion[3]*self.ring.sigma_delta**2
            cov[4,4] = sigma_0**2
            cov[5,5] = sigma_delta**2
            
        values = np.random.multivariate_normal(mean, cov, size=self.mp_number)
        self.particles["x"] = values[:,0]
        self.particles["xp"] = values[:,1]
        self.particles["y"] = values[:,2]
        self.particles["yp"] = values[:,3]
        self.particles["tau"] = values[:,4]
        self.particles["delta"] = values[:,5]
        
    def binning(self, dimension="tau", n_bin=75):
        """
        Bin macro-particles.

        Parameters
        ----------
        dimension : str, optional
            Dimension in which the binning is done. The default is "tau".
        n_bin : int, optional
            Number of bins. The default is 75.

        Returns
        -------
        bins : array of shape (n_bin,)
            Bins where the particles are sorted.
        sorted_index : array of shape (self.mp_number,)
            Bin number of each macro-particles.
        profile : array of shape (n_bin - 1,)
            Number of marco-particles in each bin.
        center : array of shape (n_bin - 1,)
            Center of each bin.

        """
        bin_min = self[dimension].min()
        bin_min = min(bin_min*0.99, bin_min*1.01)
        bin_max = self[dimension].max()
        bin_max = max(bin_max*0.99, bin_max*1.01)
        
        bins = np.linspace(bin_min, bin_max, n_bin)
        center = (bins[1:] + bins[:-1])/2
        sorted_index = np.searchsorted(bins, self[dimension], side='left')
        sorted_index -= 1
        profile = np.bincount(sorted_index, minlength=n_bin-1)
        
        return (bins, sorted_index, profile, center)
    
    def plot_profile(self, dimension="tau", n_bin=75):
        """
        Plot bunch profile.

        Parameters
        ----------
        dimension : str, optional
            Dimension to plot. The default is "tau".
        n_bin : int, optional
            Number of bins. The default is 75.

        """
        bins, sorted_index, profile, center = self.binning(dimension, n_bin)
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(center, profile)
        
    def plot_phasespace(self, x_var="tau", y_var="delta", plot_type="j"):
        """
        Plot phase space.
        
        Parameters
        ----------
        x_var : str 
            Dimension to plot on horizontal axis.
        y_var : str 
            Dimension to plot on vertical axis.
        plot_type : str {"j" , "sc"} 
            Type of the plot. The defualt value is "j" for a joint plot.
            Can be modified to "sc" for a scatter plot.
            
        Return
        ------
        fig : Figure
            Figure object with the plot on it.
        """
        
        label_dict = {"x":"x (mm)", "xp":"x' (mrad)", "y":"y (mm)", 
                      "yp":"y' (mrad)","tau":"$\\tau$ (ps)", "delta":"$\\delta$"}
        scale = {"x": 1e3, "xp":1e3, "y":1e3, "yp":1e3, "tau":1e12, "delta":1}
        
        
        if plot_type == "sc":
            fig, ax = plt.subplots()
            ax.scatter(self.particles[x_var]*scale[x_var],
                       self.particles[y_var]*scale[y_var])
            ax.set_xlabel(label_dict[x_var])
            ax.set_ylabel(label_dict[y_var])
        
        elif plot_type == "j": 
            fig = sns.jointplot(self.particles[x_var]*scale[x_var],
                                self.particles[y_var]*scale[y_var],kind="kde")
            plt.xlabel(label_dict[x_var])
            plt.ylabel(label_dict[y_var])
            
        else: 
            raise ValueError("Plot type not recognised.")
            
        return fig
        
class Beam:
    """
    Define a Beam object composed of several Bunch objects. 
    
    Parameters
    ----------
    ring : Synchrotron object
    bunch_list : list of Bunch object, optional

    Attributes
    ----------
    current : float
        Total beam current in [A]
    charge : float
        Total bunch charge in [C]
    particle_number : int
        Total number of particle in the beam
    filling_pattern : bool array of shape (ring.h,)
        Filling pattern of the beam
    bunch_current : array of shape (ring.h,)
        Current in each bunch in [A]
    bunch_charge : array of shape (ring.h,)
        Charge in each bunch in [C]
    bunch_particle : array of shape (ring.h,)
        Particle number in each bunch
    bunch_mean : array of shape (6, ring.h)
        Mean position of alive particles for each bunch
    bunch_std : array of shape (6, ring.h)
        Standard deviation of the position of alive particles for each bunch        
    bunch_emit : array of shape (6, ring.h)
        Bunch emittance of alive particles for each bunch
    mpi : Mpi object
    mpi_switch : bool
        Status of MPI parallelisation, should not be changed directly but with
        mpi_init() and mpi_close()
    bunch_index : array of shape (len(self,))
        Return an array with the positions of the non-empty bunches
        
    Methods
    ------
    init_beam(filling_pattern, current_per_bunch=1e-3, mp_per_bunch=1e3)
        Initialize beam with a given filling pattern and marco-particle number 
        per bunch. Then initialize the different bunches with a 6D gaussian
        phase space.
    mpi_init()
        Switch on MPI parallelisation and initialise a Mpi object
    mpi_gather()
        Gather beam, all bunches of the different processors are sent to 
        all processors. Rather slow
    mpi_share_distributions()
        Compute the bunch profile and share it between the different bunches.
    mpi_close()
        Call mpi_gather and switch off MPI parallelisation
    plot(var, option=None)
        Plot variables with respect to bunch number.
    """
    
    def __init__(self, ring, bunch_list=None):
        self.ring = ring
        self.mpi_switch = False
        if bunch_list is None:
            self.init_beam(np.zeros((self.ring.h,1),dtype=bool))
        else:
            if (len(bunch_list) != self.ring.h):
                raise ValueError(("The length of the bunch list is {} ".format(len(bunch_list)) + 
                                  "but should be {}".format(self.ring.h)))
            self.bunch_list = bunch_list
            
    def __len__(self):
        """Return the number of (not empty) bunches"""
        length = 0
        for bunch in self.not_empty:
            length += 1
        return length        
    
    def __getitem__(self, i):
        """Return the bunch number i"""
        return self.bunch_list.__getitem__(i)
    
    def __setitem__(self, i, value):
        """Set value to the bunch number i"""
        self.bunch_list.__setitem__(i, value)
    
    def __iter__(self):
        """Iterate over all bunches"""
        return self.bunch_list.__iter__()
   
    @property             
    def not_empty(self):
        """Return a generator to iterate over not empty bunches."""
        for index, value in enumerate(self.filling_pattern):
            if value == True:
                yield self[index]
            else:
                pass
    
    @property
    def distance_between_bunches(self):
        """Return an array which contains the distance to the next bunch in 
        units of the RF period (ring.T1)"""
        return self._distance_between_bunches
    
    def update_distance_between_bunches(self):
        """Update the distance_between_bunches array"""
        filling_pattern = self.filling_pattern
        distance = np.zeros(filling_pattern.shape)
        last_value = 0
        
        # All bunches
        for index, value in enumerate(filling_pattern):
            if value == False:
                pass
            elif value == True:
                last_value = index
                count = 1
                for value2 in filling_pattern[index+1:]:
                    if value2 == False:
                        count += 1
                    elif value2 == True:
                        break
                distance[index] = count
        
        # Last bunch case
        count2 = 0
        for index2, value2 in enumerate(filling_pattern):
            if value2 == True:
                break
            if value2 == False:
                count2 += 1
        distance[last_value] += count2
        
        self._distance_between_bunches =  distance
        
    def init_beam(self, filling_pattern, current_per_bunch=1e-3, 
                  mp_per_bunch=1e3, track_alive=True, mpi=False):
        """
        Initialize beam with a given filling pattern and marco-particle number 
        per bunch. Then initialize the different bunches with a 6D gaussian
        phase space.
        
        If the filling pattern is an array of bool then the current per bunch 
        is uniform, else the filling pattern can be an array with the current
        in each bunch.
        
        Parameters
        ----------
        filling_pattern : numpy array or list of length ring.h
            Filling pattern of the beam, can be a list or an array of bool, 
            then current_per_bunch is used. Or can be an array with the current
            in each bunch.
        current_per_bunch : float, optional
            Current per bunch in [A]
        mp_per_bunch : float, optional
            Macro-particle number per bunch
        track_alive : bool, optional
            If False, the code no longer take into account alive/dead particles.
            Should be set to True if element such as apertures are used.
            Can be set to False to gain a speed increase.
        mpi : bool, optional
            If True, only a single bunch is fully initialized on each core, the
            other bunches are initialized with a single marco-particle.
        """
        
        if (len(filling_pattern) != self.ring.h):
            raise ValueError(("The length of filling pattern is {} ".format(len(filling_pattern)) + 
                              "but should be {}".format(self.ring.h)))
        
        if mpi is True:
            mp_per_bunch_mpi = mp_per_bunch
            mp_per_bunch = 1
        
        filling_pattern = np.array(filling_pattern)
        bunch_list = []
        if filling_pattern.dtype == np.dtype("bool"):
            for value in filling_pattern:
                if value == True:
                    bunch_list.append(Bunch(self.ring, mp_per_bunch, 
                                            current_per_bunch, track_alive))
                elif value == False:
                    bunch_list.append(Bunch(self.ring, alive=False))
        elif filling_pattern.dtype == np.dtype("float64"):
            for current in filling_pattern:
                if current != 0:
                    bunch_list.append(Bunch(self.ring, mp_per_bunch, 
                                            current, track_alive))
                elif current == 0:
                    bunch_list.append(Bunch(self.ring, alive=False))
        else:
            raise TypeError("{} should be bool or float64".format(filling_pattern.dtype))
                
        self.bunch_list = bunch_list
        self.update_filling_pattern()
        self.update_distance_between_bunches()
        
        if mpi is True:
            self.mpi_init()
            current = self[self.mpi.rank_to_bunch(self.mpi.rank)].current
            bunch =  Bunch(self.ring, mp_per_bunch_mpi, current, track_alive)
            bunch.init_gaussian()
            self[self.mpi.rank_to_bunch(self.mpi.rank)] = bunch
        else:
            for bunch in self.not_empty:
                bunch.init_gaussian()
    
    def update_filling_pattern(self):
        """Update the beam filling pattern."""
        filling_pattern = []
        for bunch in self:
            if bunch.current != 0:
                filling_pattern.append(True)
            else:
                filling_pattern.append(False)
        self._filling_pattern = np.array(filling_pattern)
    
    @property
    def filling_pattern(self):
        """Return an array with the filling pattern of the beam as bool"""
        return self._filling_pattern
    
    @property
    def bunch_index(self):
        """Return an array with the positions of the non-empty bunches."""
        return np.where(self.filling_pattern == True)[0]
        
    @property
    def bunch_current(self):
        """Return an array with the current in each bunch in [A]"""
        bunch_current = [bunch.current for bunch in self]
        return np.array(bunch_current)
    
    @property
    def bunch_charge(self):
        """Return an array with the charge in each bunch in [C]"""
        bunch_charge = [bunch.charge for bunch in self]
        return np.array(bunch_charge)
    
    @property
    def bunch_particle(self):
        """Return an array with the particle number in each bunch"""
        bunch_particle = [bunch.particle_number for bunch in self]
        return np.array(bunch_particle)
    
    @property
    def current(self):
        """Total beam current in [A]"""
        return np.sum(self.bunch_current)
    
    @property
    def charge(self):
        """Total beam charge in [C]"""
        return np.sum(self.bunch_charge)
    
    @property
    def particle_number(self):
        """Total number of particles in the beam"""
        return np.sum(self.bunch_particle)
    
    @property
    def bunch_mean(self):
        """Return an array with the mean position of alive particles for each
        bunches"""
        bunch_mean = np.zeros((6,self.ring.h))
        for idx, bunch in enumerate(self.not_empty):
            index = self.bunch_index[idx]
            bunch_mean[:,index] = bunch.mean
        return bunch_mean
    
    @property
    def bunch_std(self):
        """Return an array with the standard deviation of the position of alive 
        particles for each bunches"""
        bunch_std = np.zeros((6,self.ring.h))
        for idx, bunch in enumerate(self.not_empty):
            index = self.bunch_index[idx]
            bunch_std[:,index] = bunch.std
        return bunch_std
    
    @property
    def bunch_emit(self):
        """Return an array with the bunch emittance of alive particles for each
        bunches and each plane"""
        bunch_emit = np.zeros((3,self.ring.h))
        for idx, bunch in enumerate(self.not_empty):
            index = self.bunch_index[idx]
            bunch_emit[:,index] = bunch.emit
        return bunch_emit
    
    @property
    def bunch_cs(self):
        """Return an array with the average Courant-Snyder invariant for each 
        bunch"""
        bunch_cs = np.zeros((2,self.ring.h))
        for idx, bunch in enumerate(self.not_empty):
            index = self.bunch_index[idx]
            bunch_cs[:,index] = bunch.cs_invariant
        return bunch_cs
    
    def mpi_init(self):
        """Switch on MPI parallelisation and initialise a Mpi object"""
        from mbtrack2.tracking.parallel import Mpi
        self.mpi = Mpi(self.filling_pattern)
        self.mpi_switch = True
        
    def mpi_gather(self):
        """Gather beam, all bunches of the different processors are sent to 
        all processors. Rather slow"""
        
        if(self.mpi_switch == False):
            print("Error, mpi is not initialised.")
        
        bunch = self[self.mpi.bunch_num]
        bunches = self.mpi.comm.allgather(bunch)
        for rank in range(self.mpi.size):
            self[self.mpi.rank_to_bunch(rank)] = bunches[rank]
            
    def mpi_close(self):
        """Call mpi_gather and switch off MPI parallelisation"""
        self.mpi_gather()
        self.mpi_switch = False
        self.mpi = None
        
    def plot(self, var, option=None):
        """
        Plot variables with respect to bunch number.

        Parameters
        ----------
        var : str {"bunch_current", "bunch_charge", "bunch_particle", 
                   "bunch_mean", "bunch_std", "bunch_emit"}
            Variable to be plotted.
        option : str, optional
            If var is "bunch_mean", "bunch_std", or "bunch_emit, option needs 
            to be specified.
            For "bunch_mean" and "bunch_std", 
                option = {"x","xp","y","yp","tau","delta"}.
            For "bunch_emit", option = {"x","y","s"}.
            The default is None.
            
        Return
        ------
        fig : Figure
            Figure object with the plot on it.
        """
        
        var_dict = {"bunch_current":self.bunch_current,
                    "bunch_charge":self.bunch_charge,
                    "bunch_particle":self.bunch_particle,
                    "bunch_mean":self.bunch_mean,
                    "bunch_std":self.bunch_std,
                    "bunch_emit":self.bunch_emit}
        
        fig, ax= plt.subplots()
        
        if var == "bunch_mean" or var == "bunch_std":
            value_dict = {"x":0, "xp":1, "y":2, "yp":3, "tau":4, "delta":5}
            scale = [1e6, 1e6, 1e6, 1e6, 1e12, 1]
            label_mean = ["x (um)", "x' ($\\mu$rad)", "y (um)", "y' ($\\mu$rad)",
                      "$\\tau$ (ps)", "$\\delta$"]
            label_std = ["std x (um)", "std x' ($\\mu$rad)", "std y (um)",
                        "std y' ($\\mu$rad)", "std $\\tau$ (ps)",
                        "std $\\delta$"]
           
            y_axis = var_dict[var][value_dict[option]]
            
            # Convert NaN in y_axis array into zero
            where_is_nan = np.isnan(y_axis)
            y_axis[where_is_nan] = 0
            
            ax.plot(np.arange(len(self.filling_pattern)),
                      y_axis*scale[value_dict[option]])
            ax.set_xlabel('bunch number')
            if var == "bunch_mean":
                ax.set_ylabel(label_mean[value_dict[option]])
            else: 
                ax.set_ylabel(label_std[value_dict[option]])
            
        elif var == "bunch_emit":
            value_dict = {"x":0, "y":1, "s":2}
            scale = [1e9, 1e9, 1e15]
            
            y_axis = var_dict[var][value_dict[option]]
            
            # Convert NaN in y_axis array into zero
            where_is_nan = np.isnan(y_axis)
            y_axis[where_is_nan] = 0
            
            ax.plot(np.arange(len(self.filling_pattern)), 
                     y_axis*scale[value_dict[option]])
            
            if option == "x": label_y = "hor. emittance (nm.rad)"
            elif option == "y": label_y = "ver. emittance (nm.rad)"
            elif option == "s": label_y =  "long. emittance (fm.rad)"
            
            ax.set_xlabel('bunch number')
            ax.set_ylabel(label_y)
                
        elif var=="bunch_current" or var=="bunch_charge" or var=="bunch_particle":
            scale = {"bunch_current":1e3, "bunch_charge":1e9, 
                     "bunch_particle":1}
            
            ax.plot(np.arange(len(self.filling_pattern)), var_dict[var]*
                     scale[var]) 
            ax.set_xlabel('bunch number')
            
            if var == "bunch_current": label_y = "bunch current (mA)"
            elif var == "bunch_charge": label_y = "bunch chagre (nC)"
            else: label_y = "number of particles"

            ax.set_ylabel(label_y)             
    
        elif var == "current" or var=="charge" or var=="particle_number":
            raise ValueError("'{0}'is a total value and cannot be plotted."
                             .format(var))
       
        return fig
        
    
