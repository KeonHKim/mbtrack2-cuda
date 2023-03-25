# -*- coding: utf-8 -*-
"""
This module defines aperture elements for tracking.
"""

import numpy as np
from mbtrack2.tracking.element import Element

class CircularAperture(Element):
    """
    Circular aperture element. The particles which are outside of the circle 
    are 'lost' and not used in the tracking any more.
    
    Parameters
    ----------
    radius : float
        radius of the circle in [m]
    """
    
    def __init__(self, radius):
        self.radius = radius
        self.radius_squared = radius**2
    
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
        alive = bunch.particles["x"]**2 + bunch.particles["y"]**2 < self.radius_squared
        bunch.alive[~alive] = False
        
class ElipticalAperture(Element):
    """
    Eliptical aperture element. The particles which are outside of the elipse 
    are 'lost' and not used in the tracking any more.
    
    Parameters
    ----------
    X_radius : float
        horizontal radius of the elipse in [m]
    Y_radius : float
        vertical radius of the elipse in [m]
    """
    
    def __init__(self, X_radius, Y_radius):
        self.X_radius = X_radius
        self.X_radius_squared = X_radius**2
        self.Y_radius = Y_radius
        self.Y_radius_squared =Y_radius**2
    
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
        alive = (bunch.particles["x"]**2/self.X_radius_squared + 
                 bunch.particles["y"]**2/self.Y_radius_squared < 1)
        bunch.alive[~alive] = False

class RectangularAperture(Element):
    """
    Rectangular aperture element. The particles which are outside of the 
    rectangle are 'lost' and not used in the tracking any more.
    
    Parameters
    ----------
    X_right : float
        right horizontal aperture of the rectangle in [m]
    Y_top : float
        top vertical aperture of the rectangle in [m]
    X_left : float, optional
        left horizontal aperture of the rectangle in [m]
    Y_bottom : float, optional
        bottom vertical aperture of the rectangle in [m]
    """
    
    def __init__(self, X_right, Y_top, X_left=None, Y_bottom=None):
        self.X_right = X_right
        self.X_left = X_left
        self.Y_top = Y_top
        self.Y_bottom = Y_bottom
 
    
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
        
        if (self.X_left is None):
            alive_X = np.abs(bunch.particles["x"]) < self.X_right
        else:
            alive_X = ((bunch.particles["x"] < self.X_right) &
                     (bunch.particles["x"] > self.X_left))
            
        if (self.Y_bottom is None):
            alive_Y = np.abs(bunch.particles["y"]) < self.Y_top
        else:
            alive_Y = ((bunch.particles["y"] < self.Y_top) &
                     (bunch.particles["y"] > self.Y_bottom))

        alive = alive_X & alive_Y
        bunch.alive[~alive] = False
        
class LongitudinalAperture(Element):
    """
    Longitudinal aperture element. The particles which are outside of the 
    longitudinal bounds are 'lost' and not used in the tracking any more.
    
    Parameters
    ----------
    ring : Synchrotron object
    tau_up : float
        Upper longitudinal bound in [s].
    tau_low : float, optional
        Lower longitudinal bound in [s].
    """
    
    def __init__(self, tau_up, tau_low=None):
        self.tau_up = tau_up
        if tau_low is None:
            self.tau_low = tau_up*-1
        else:
            self.tau_low = tau_low
    
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
        
        alive = ((bunch.particles["tau"] < self.tau_up) &
                 (bunch.particles["tau"] > self.tau_low))

        bunch.alive[~alive] = False