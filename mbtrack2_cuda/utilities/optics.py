# -*- coding: utf-8 -*-
"""
Module where the class to store the optic functions and the lattice physical 
parameters are defined.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Optics:
    """
    Class used to handle optic functions.
    
    Parameters
    ----------
    lattice_file : str, optional if local_beta, local_alpha and 
        local_dispersion are specified.
        AT lattice file path.
    local_beta : array of shape (2,), optional if lattice_file is specified.
        Beta function at the location of the tracking. Default is mean beta if
        lattice has been loaded.
    local_alpha : array of shape (2,), optional if lattice_file is specified.
        Alpha function at the location of the tracking. Default is mean alpha 
        if lattice has been loaded.
    local_dispersion : array of shape (4,), optional if lattice_file is 
        specified.
        Dispersion function and derivative at the location of the tracking. 
        Default is zero if lattice has been loaded.  
        
    Attributes
    ----------
    use_local_values : bool
        True if no lattice has been loaded.
    local_gamma : array of shape (2,)
        Gamma function at the location of the tracking.
    lattice : AT lattice
        
    Methods
    -------
    load_from_AT(lattice_file, **kwargs)
        Load a lattice from accelerator toolbox (AT).
    setup_interpolation()
        Setup interpolation of the optic functions.
    beta(position)
        Return beta functions at specific locations given by position.
    alpha(position)
        Return alpha functions at specific locations given by position.
    gamma(position)
        Return gamma functions at specific locations given by position.
    dispersion(position)
        Return dispersion functions at specific locations given by position.
    plot(self, var, option, n_points=1000)
        Plot optical variables.
    """
    
    def __init__(self, lattice_file=None, local_beta=None, local_alpha=None, 
                 local_dispersion=None, **kwargs):
        
        if lattice_file is not None:
            self.use_local_values = False
            self.load_from_AT(lattice_file, **kwargs)
            if local_beta is None:
                self._local_beta = np.mean(self.beta_array, axis=1)
            else:
                self._local_beta = local_beta
            if local_alpha is None:
                self._local_alpha = np.mean(self.alpha_array, axis=1)
            else:
                self._local_alpha = local_alpha
            if local_dispersion is None:
                self.local_dispersion = np.zeros((4,))
            else:
                self.local_dispersion = local_dispersion
            self._local_gamma = (1 + self._local_alpha**2)/self._local_beta
            
        else:
            self.use_local_values = True
            self._local_beta = local_beta
            self._local_alpha = local_alpha
            self._local_gamma = (1 + self._local_alpha**2)/self._local_beta
            self.local_dispersion = local_dispersion

    def load_from_AT(self, lattice_file, **kwargs):
        """
        Load a lattice from accelerator toolbox (AT).

        Parameters
        ----------
        lattice_file : str
            AT lattice file path.
        n_points : int or float, optional
            Minimum number of points to use for the optic function arrays.
        periodicity : int, optional
            Lattice periodicity, if not specified the AT lattice periodicity is
            used.
        """
        import at
        self.n_points = int(kwargs.get("n_points", 1e3))
        periodicity = kwargs.get("periodicity")
        
        self.lattice = at.load_lattice(lattice_file)
        if self.lattice.radiation:
            self.lattice.radiation_off()
        lattice = self.lattice.slice(slices=self.n_points)
        refpts = np.arange(0, len(lattice))
        twiss0, tune, chrom, twiss = at.linopt(lattice, refpts=refpts,
                                                  get_chrom=True)
        
        if periodicity is None:
            self.periodicity = lattice.periodicity
        else:
            self.periodicity = periodicity
        
        if self.periodicity > 1:
            for i in range(self.periodicity-1):
                pos = np.append(twiss.s_pos, twiss.s_pos + twiss.s_pos[-1]*(i+1))
        else:
            pos = twiss.s_pos
            
        self.position = pos
        self.beta_array = np.tile(twiss.beta.T, self.periodicity)
        self.alpha_array = np.tile(twiss.alpha.T, self.periodicity)
        self.dispersion_array = np.tile(twiss.dispersion.T, self.periodicity)
        
        self.position = np.append(self.position, self.lattice.circumference)
        self.beta_array = np.append(self.beta_array, self.beta_array[:,0:1],
                                    axis=1)
        self.alpha_array = np.append(self.alpha_array, self.alpha_array[:,0:1],
                                     axis=1)
        self.dispersion_array = np.append(self.dispersion_array,
                                          self.dispersion_array[:,0:1], axis=1)
        
        self.gamma_array = (1 + self.alpha_array**2)/self.beta_array
        self.tune = tune * self.periodicity
        self.chro = chrom * self.periodicity
        self.ac = at.get_mcf(self.lattice)
        
        self.setup_interpolation()
        
        
    def setup_interpolation(self):      
        """Setup interpolation of the optic functions."""
        self.betaX = interp1d(self.position, self.beta_array[0,:],
                              kind='linear')
        self.betaY = interp1d(self.position, self.beta_array[1,:],
                              kind='linear')
        self.alphaX = interp1d(self.position, self.alpha_array[0,:],
                               kind='linear')
        self.alphaY = interp1d(self.position, self.alpha_array[1,:],
                               kind='linear')
        self.gammaX = interp1d(self.position, self.gamma_array[0,:],
                               kind='linear')
        self.gammaY = interp1d(self.position, self.gamma_array[1,:],
                               kind='linear')
        self.dispX = interp1d(self.position, self.dispersion_array[0,:],
                              kind='linear')
        self.disppX = interp1d(self.position, self.dispersion_array[1,:],
                               kind='linear')
        self.dispY = interp1d(self.position, self.dispersion_array[2,:],
                              kind='linear')
        self.disppY = interp1d(self.position, self.dispersion_array[3,:],
                               kind='linear')
    
    @property
    def local_beta(self):
        """
        Return beta function at the location defined by the lattice file.

        """
        return self._local_beta
    
    @local_beta.setter
    def local_beta(self, beta_array):
        """
        Set the values of beta function. Gamma function is automatically 
        recalculated after the new value of beta function is set.

        Parameters
        ----------
        beta_array : array of shape (2,)
            Beta function in the horizontal and vertical plane.

        """
        self._local_beta = beta_array
        self._local_gamma = (1+self._local_alpha**2) / self._local_beta
        
    @property
    def local_alpha(self):
        """
        Return alpha function at the location defined by the lattice file.

        """
        return self._local_alpha
    
    @local_alpha.setter
    def local_alpha(self, alpha_array):
        """
        Set the value of beta functions. Gamma function is automatically 
        recalculated after the new value of alpha function is set.

        Parameters
        ----------
        alpha_array : array of shape (2,)
            Alpha function in the horizontal and vertical plane.

        """
        self._local_alpha = alpha_array
        self._local_gamma = (1+self._local_alpha**2) / self._local_beta
    
    @property
    def local_gamma(self):
        """
        Return beta function at the location defined by the lattice file.

        """
        return self._local_gamma
    
    def beta(self, position):
        """
        Return beta functions at specific locations given by position. If no
        lattice has been loaded, local values are returned.

        Parameters
        ----------
        position : array or float
            Longitudinal position at which the beta functions are returned.

        Returns
        -------
        beta : array
            Beta functions.
        """
        if self.use_local_values:
            return np.outer(self.local_beta, np.ones_like(position))
        else:
            beta = [self.betaX(position), self.betaY(position)]
            return np.array(beta)
    
    def alpha(self, position):
        """
        Return alpha functions at specific locations given by position. If no
        lattice has been loaded, local values are returned.

        Parameters
        ----------
        position : array or float
            Longitudinal position at which the alpha functions are returned.

        Returns
        -------
        alpha : array
            Alpha functions.
        """
        if self.use_local_values:
            return np.outer(self.local_alpha, np.ones_like(position))
        else:
            alpha = [self.alphaX(position), self.alphaY(position)]
            return np.array(alpha)
    
    def gamma(self, position):
        """
        Return gamma functions at specific locations given by position. If no
        lattice has been loaded, local values are returned.

        Parameters
        ----------
        position : array or float
            Longitudinal position at which the gamma functions are returned.

        Returns
        -------
        gamma : array
            Gamma functions.
        """
        if self.use_local_values:
            return np.outer(self.local_gamma, np.ones_like(position))
        else:
            gamma = [self.gammaX(position), self.gammaY(position)]
            return np.array(gamma)
    
    def dispersion(self, position):
        """
        Return dispersion functions at specific locations given by position. 
        If no lattice has been loaded, local values are returned.

        Parameters
        ----------
        position : array or float
            Longitudinal position at which the dispersion functions are 
            returned.

        Returns
        -------
        dispersion : array
            Dispersion functions.
        """
        if self.use_local_values:
            return np.outer(self.local_dispersion, np.ones_like(position))
        else:
            dispersion = [self.dispX(position), self.disppX(position), 
                          self.dispY(position), self.disppY(position)]
            return np.array(dispersion)
        
    def plot(self, var, option, n_points=1000):
        """
        Plot optical variables.
    
        Parameters
        ----------
        var : {"beta", "alpha", "gamma", "dispersion"}
            Optical variable to be plotted.
        option : str
            If var = "beta", "alpha" and "gamma", option = {"x","y"} specifying
            the axis of interest.
            If var = "dispersion", option = {"x","px","y","py"} specifying the 
            axis of interest for the dispersion function or its derivative.
        n_points : int
            Number of points on the plot. The default value is 1000.
    
        """
    
        var_dict = {"beta":self.beta, "alpha":self.alpha, "gamma":self.gamma, 
                    "dispersion":self.dispersion}
        
        if var == "dispersion":
            option_dict = {"x":0, "px":1, "y":2, "py":3}
            
            label = ["D$_{x}$ (m)", "D'$_{x}$", "D$_{y}$ (m)", "D'$_{y}$"]
            
            ylabel = label[option_dict[option]]
         
        
        elif var=="beta" or var=="alpha" or var=="gamma":
            option_dict = {"x":0, "y":1}
            label_dict = {"beta":"$\\beta$", "alpha":"$\\alpha$", 
                          "gamma":"$\\gamma$"}
            
            if option == "x": label_sup = "$_{x}$"
            elif option == "y": label_sup = "$_{y}$"
            
            unit = {"beta":" (m)", "alpha":"", "gamma":" (m$^{-1}$)"}
            
            ylabel = label_dict[var] + label_sup + unit[var]
  
                
        else:
            raise ValueError("Variable name is not found.")
        
        if self.use_local_values is not True:
            position = np.linspace(0, self.lattice.circumference, int(n_points))
        else: 
            position = np.linspace(0,1)
            
        var_list = var_dict[var](position)[option_dict[option]]
        fig, ax = plt.subplots()
        ax.plot(position,var_list)
           
        ax.set_xlabel("position (m)")
        ax.set_ylabel(ylabel)
        
        return fig

    
class PhysicalModel:
    """
    Store the lattice physical parameters such as apperture and resistivity.
    
    Parameters
    ----------
    ring : Synchrotron object
    x_right : float
        Default value for the right horizontal aperture in [m].
    y_top : float
        Default value for the top vertical aperture in [m].
    shape : str
        Default value for the shape of the chamber cross section 
        (elli/circ/rect).
    rho : float
        Default value for the resistivity of the chamber material [ohm.m].
    x_left : float, optional
        Default value for the left horizontal aperture in [m].
    y_bottom : float, optional
        Default value for the bottom vertical aperture in [m]. 
    n_points : int or float, optional
        Number of points to use in class arrays
        
    Attributes
    ----------
    position : array of shape (n_points,)
        Longitudinal position in [m].
    x_right : array of shape (n_points,)
        Right horizontal aperture in [m].
    y_top : array of shape (n_points,)
        Top vertical aperture in [m].
    shape : array of shape (n_points - 1,)
        Shape of the chamber cross section (elli/circ/rect).
    rho : array of shape (n_points - 1,)
        Resistivity of the chamber material in [ohm.m].
    x_left : array of shape (n_points,)
        Left horizontal aperture in [m].
    y_bottom : array of shape (n_points,)
        Bottom vertical aperture in [m].
    length : array of shape (n_points - 1,)
        Length of each segments between two longitudinal positions in [m].
    center : array of shape (n_points - 1,)
        Center of each segments between two longitudinal positions in [m].
        
    Methods
    -------
    change_values(start_position, end_position, x_right, y_top, shape, rho)
        Change the physical parameters between start_position and end_position.
    taper(start_position, end_position, x_right_start, x_right_end, 
          y_top_start, y_top_end, shape, rho)
        Change the physical parameters to have a tapered transition between 
        start_position and end_position.
    plot_aperture()
        Plot horizontal and vertical apertures.
    resistive_wall_effective_radius(optics)
        Return the effective radius of the chamber for resistive wall 
        calculations.
    """
    def __init__(self, ring, x_right, y_top, shape, rho, x_left=None, 
                 y_bottom=None, n_points=1e4):
        
        self.n_points = int(n_points)
        self.position = np.linspace(0, ring.L, self.n_points)
        self.x_right = np.ones_like(self.position)*x_right 
        self.y_top = np.ones_like(self.position)*y_top
        
        if x_left is None:
            self.x_left = np.ones_like(self.position)*-1*x_right
        else:
            self.x_left = np.ones_like(self.position)*x_left
        
        if y_bottom is None:
            self.y_bottom = np.ones_like(self.position)*-1*y_top
        else:
            self.y_bottom = np.ones_like(self.position)*y_bottom
        
        self.length = self.position[1:] - self.position[:-1]
        self.center = (self.position[1:] + self.position[:-1])/2
        self.rho = np.ones_like(self.center)*rho
        
        self.shape = np.repeat(np.array([shape]), self.n_points-1)

    def resistive_wall_effective_radius(self, optics, x_right=True, 
                                        y_top=True):
        """
        Return the effective radius of the chamber for resistive wall 
        calculations as defined in Eq. 27 of [1].

        Parameters
        ----------
        optics : Optics object
        x_right : bool, optional
            If True, x_right is used, if Fasle, x_left is used.
        y_top : TYPE, optional
            If True, y_top is used, if Fasle, y_bottom is used.

        Returns
        -------
        rho_star : float
            Effective resistivity of the chamber material in [ohm.m].
        a1_L : float
            Effective longitudinal radius of the chamber of the first power in
            [m].
        a2_L : float
            Effective longitudinal radius of the chamber of the second power 
            in [m].
        a3_H : float
            Effective horizontal radius of the chamber of the third power in
            [m].
        a4_H : float
            Effective horizontal radius of the chamber of the fourth power in
            [m].
        a3_V : float
            Effective vertical radius of the chamber of the third power in [m].
        a4_V : float
            Effective vertical radius of the chamber of the fourth power in 
            [m].

        References  
        ----------
        [1] Skripka, Galina, et al. "Simultaneous computation of intrabunch 
        and interbunch collective beam motions in storage rings." Nuclear 
        Instruments and Methods in Physics Research Section A: Accelerators, 
        Spectrometers, Detectors and Associated Equipment 806 (2016): 221-230.         
        """
        
        if x_right is True:
            a0 = (self.x_right[1:] + self.x_right[:-1])/2
        else:
            a0 = np.abs((self.x_left[1:] + self.x_left[:-1])/2)
            
        if y_top is True:
            b0 = (self.y_top[1:] + self.y_top[:-1])/2
        else:
            b0 = np.abs((self.y_bottom[1:] + self.y_bottom[:-1])/2)
        
        beta = optics.beta(self.center)
        L = self.position[-1]
        sigma = 1/self.rho
        beta_H_star = 1/L*(self.length*beta[0,:]).sum()
        beta_V_star = 1/L*(self.length*beta[1,:]).sum()
        sigma_star = 1/L*(self.length*sigma).sum()
        
        a1_H = (((self.length*beta[0,:]/(np.sqrt(sigma)*(a0)**1)).sum())**(-1)*L*beta_H_star/np.sqrt(sigma_star))**(1/1)
        a2_H = (((self.length*beta[0,:]/(np.sqrt(sigma)*(a0)**2)).sum())**(-1)*L*beta_H_star/np.sqrt(sigma_star))**(1/2)

        a1_V = (((self.length*beta[1,:]/(np.sqrt(sigma)*(b0)**1)).sum())**(-1)*L*beta_V_star/np.sqrt(sigma_star))**(1/1)
        a2_V = (((self.length*beta[1,:]/(np.sqrt(sigma)*(b0)**2)).sum())**(-1)*L*beta_V_star/np.sqrt(sigma_star))**(1/2)
        
        a3_H = (((self.length*beta[0,:]/(np.sqrt(sigma)*(a0)**3)).sum())**(-1)*L*beta_H_star/np.sqrt(sigma_star))**(1/3)
        a4_H = (((self.length*beta[0,:]/(np.sqrt(sigma)*(a0)**4)).sum())**(-1)*L*beta_H_star/np.sqrt(sigma_star))**(1/4)
        
        a3_V = (((self.length*beta[1,:]/(np.sqrt(sigma)*(b0)**3)).sum())**(-1)*L*beta_V_star/np.sqrt(sigma_star))**(1/3)
        a4_V = (((self.length*beta[1,:]/(np.sqrt(sigma)*(b0)**4)).sum())**(-1)*L*beta_V_star/np.sqrt(sigma_star))**(1/4)
        
        a1_L = min((a1_H,a1_V))
        a2_L = min((a2_H,a2_V))
        
        return (1/sigma_star, a1_L, a2_L, a3_H, a4_H, a3_V, a4_V)
        
    def change_values(self, start_position, end_position, x_right, y_top, 
                      shape, rho, x_left=None, y_bottom=None):
        """
        Change the physical parameters between start_position and end_position.

        Parameters
        ----------
        start_position : float
        end_position : float
        x_right : float
            Right horizontal aperture in [m].
        y_top : float
            Top vertical aperture in [m].
        shape : str
            Shape of the chamber cross section (elli/circ/rect).
        rho : float
            Resistivity of the chamber material [ohm.m].
        x_left : float, optional
            Left horizontal aperture in [m].
        y_bottom : float, optional
            Bottom vertical aperture in [m].
        """
        ind = (self.position > start_position) & (self.position < end_position)
        self.x_right[ind] = x_right
        self.y_top[ind] = y_top
        
        if x_left is None:
            self.x_left[ind] = -1*x_right
        else:
            self.x_left[ind] = x_left
        
        if y_bottom is None:
            self.y_bottom[ind] = -1*y_top
        else:
            self.y_bottom[ind] = y_bottom
        
        ind2 = ((self.position[:-1] > start_position) & 
                (self.position[1:] < end_position))
        self.rho[ind2] = rho
        self.shape[ind2] = shape
        
    def taper(self, start_position, end_position, x_right_start, x_right_end,
              y_top_start, y_top_end, shape, rho, x_left_start=None, 
              x_left_end=None, y_bottom_start=None, y_bottom_end=None):
        """
        Change the physical parameters to have a tapered transition between 
        start_position and end_position.

        Parameters
        ----------
        start_position : float
        end_position : float
        x_right_start : float
            Right horizontal aperture at the start of the taper in [m].
        x_right_end : float
            Right horizontal aperture at the end of the taper in [m].
        y_top_start : float
            Top vertical aperture at the start of the taper in [m].
        y_top_end : float
            Top vertical aperture at the end of the taper in [m].
        shape : str
            Shape of the chamber cross section (elli/circ/rect).
        rho : float
            Resistivity of the chamber material [ohm.m].
        x_left_start : float, optional
            Left horizontal aperture at the start of the taper in [m].
        x_left_end : float, optional
            Left horizontal aperture at the end of the taper in [m].
        y_bottom_start : float, optional
            Bottom vertical aperture at the start of the taper in [m].
        y_bottom_end : float, optional
            Bottom vertical aperture at the end of the taper in [m].
        """
        ind = (self.position > start_position) & (self.position < end_position)
        self.x_right[ind] = np.linspace(x_right_start, x_right_end, sum(ind))
        self.y_top[ind] = np.linspace(y_top_start, y_top_end, sum(ind))
        
        if (x_left_start is not None) and (x_left_end is not None):
            self.x_left[ind] = np.linspace(x_left_start, x_left_end, sum(ind))
        else:
            self.x_left[ind] = -1*np.linspace(x_right_start, x_right_end, 
                                              sum(ind))
            
        if (y_bottom_start is not None) and (y_bottom_end is not None):
            self.y_bottom[ind] = np.linspace(y_bottom_start, y_bottom_end, 
                                             sum(ind))
        else:
            self.y_bottom[ind] = -1*np.linspace(y_top_start, y_top_end, 
                                                sum(ind))
        
        ind2 = ((self.position[:-1] > start_position) 
                & (self.position[1:] < end_position))
        self.rho[ind2] = rho
        self.shape[ind2] = shape
        
    def plot_aperture(self):
        """Plot horizontal and vertical apertures."""
        fig, axs = plt.subplots(2)
        axs[0].plot(self.position,self.x_right*1e3)
        axs[0].plot(self.position,self.x_left*1e3)
        axs[0].set(xlabel="Longitudinal position [m]", 
                   ylabel="Horizontal aperture [mm]")
        axs[0].legend(["Right","Left"])
        
        axs[1].plot(self.position,self.y_top*1e3)
        axs[1].plot(self.position,self.y_bottom*1e3)
        axs[1].set(xlabel="Longitudinal position [m]", 
                   ylabel="Vertical aperture [mm]")
        axs[1].legend(["Top","Bottom"])
        
