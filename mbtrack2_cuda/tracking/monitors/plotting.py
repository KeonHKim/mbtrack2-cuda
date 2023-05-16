# -*- coding: utf-8 -*-
"""
Module for plotting the data recorded by the monitor module during the 
tracking.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import h5py as hp
import random
from scipy.stats import gmean

def plot_beamdata(filenames, dataset="mean", dimension="tau", stat_var="mean", 
                  x_var="time", turn=None, legend=None):
    """
    Plot 2D data recorded by BeamMonitor.

    Parameters
    ----------
    filenames : str or list of str
        Names of the HDF5 files to be plotted.
    dataset : {"current","emit","mean","std","cs_invariant"}
        HDF5 file's dataset to be plotted. The default is "mean".
    dimension : str
         The dimension of the dataset to plot:
            for "emit", dimension = {"x","y","s"},
            for "cs_invariant", dimension = {"x","y"},
            for "mean" and "std", dimension = {"x","xp","y","yp","tau","delta"}.
            not used if "current".
        The default is "tau".
    stat_var : {"mean", "std"}
        Statistical value of the dimension.
    x_var : {"time", "index"}
        Choice of the horizontal axis:
            "time" corresponds to turn number.
            "index" corresponds to bunch index.
    turn : int or float, optional
        Choice of the turn to plot when x_var = "index".
        If None, the last turn is plotted.
    legend : list of str, optional
        Legend to add for each file.

    Return
    ------
    fig : Figure
        Figure object with the plot on it.

    """
    
    if isinstance(filenames, str):
        filenames = [filenames]
    
    fig, ax = plt.subplots()
    
    for filename in filenames:
        file = hp.File(filename, "r")
        time = np.array(file["Beam"]["time"])
        data = np.array(file["Beam"][dataset])
            
        if x_var == "time":
            x = time
            x_label = "Number of turns"
            bunch_index = (file["Beam"]["current"][:,0] != 0).nonzero()[0]
            if dataset == "current":
                y = np.nansum(data[bunch_index,:],0)*1e3
                y_label = "Total current (mA)"
            elif dataset == "emit":
                dimension_dict = {"x":0, "y":1, "s":2}
                axis = dimension_dict[dimension]
                label = ["$\\epsilon_{x}$ (m.rad)",
                         "$\\epsilon_{y}$ (m.rad)",
                         "$\\epsilon_{s}$ (s)"]
                if stat_var == "mean":
                    y = np.nanmean(data[axis,bunch_index,:],0)
                elif stat_var == "std":
                    y = np.nanstd(data[axis,bunch_index,:],0)
                y_label = stat_var + " " + label[axis]
            elif dataset == "cs_invariant":
                dimension_dict = {"x":0, "y":1}
                axis = dimension_dict[dimension]
                label = ['$J_x$ (m)', '$J_y$ (m)']
                if stat_var == "mean":
                    y = np.nanmean(data[axis,bunch_index,:],0)
                elif stat_var == "std":
                    y = np.nanstd(data[axis,bunch_index,:],0)
                y_label = stat_var + " " + label[axis]
            elif dataset == "mean" or dataset == "std":
                dimension_dict = {"x":0, "xp":1, "y":2, "yp":3, "tau":4, 
                                  "delta":5}
                axis = dimension_dict[dimension]
                scale = [1e6, 1e6, 1e6, 1e6, 1e12, 1]
                label = ["x (um)", "x' ($\\mu$rad)", "y (um)", 
                         "y' ($\\mu$rad)", "$\\tau$ (ps)", "$\\delta$"]
                if stat_var == "mean":   
                    y = np.nanmean(data[axis,bunch_index,:],0)*scale[axis]
                elif stat_var == "std":      
                    y = np.nanstd(data[axis,bunch_index,:],0)*scale[axis]
                label_sup = {"mean":"mean of ", "std":"std of "}
                y_label = label_sup[stat_var] + dataset + " " + label[axis]
                
        elif x_var == "index":
            h = len(file["Beam"]["mean"][0,:,0])
            x = np.arange(h)
            x_label = "Bunch index"
            if turn is None:
                idx = -1
            else:
                idx = np.where(time == int(turn))[0]
                if (idx.size == 0):
                    raise ValueError("Turn is not valid.")
            
            if dataset == "current":
                y = data[:,idx]*1e3
                y_label = "Bunch current (mA)"
            elif dataset == "emit":
                dimension_dict = {"x":0, "y":1, "s":2}
                axis = dimension_dict[dimension]
                label = ["$\\epsilon_{x}$ (m.rad)",
                         "$\\epsilon_{y}$ (m.rad)",
                         "$\\epsilon_{s}$ (s)"]
                y = data[axis,:,idx]
                y_label = label[axis]
            elif dataset == "cs_invariant":
                dimension_dict = {"x":0, "y":1}
                axis = dimension_dict[dimension]
                label = ['$J_x$ (m)', '$J_y$ (m)']
                y = data[axis,:,idx]
                y_label = label[axis]
            elif dataset == "mean" or dataset == "std":
                dimension_dict = {"x":0, "xp":1, "y":2, "yp":3, "tau":4, 
                                  "delta":5}
                axis = dimension_dict[dimension]
                scale = [1e6, 1e6, 1e6, 1e6, 1e12, 1]
                label = ["x (um)", "x' ($\\mu$rad)", "y (um)", 
                         "y' ($\\mu$rad)", "$\\tau$ (ps)", "$\\delta$"]
                y = data[axis,:,idx]*scale[axis]
                y_label = dataset + " " + label[axis]
        else:
            raise ValueError("x_var should be time or index")
            
        y = np.squeeze(y)
        
        ax.plot(x, y)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if legend is not None:
            plt.legend(legend)
            
        file.close()
        
    return fig
            
def streak_beamdata(filename, dataset="mean", dimension="tau", cm_lim=None):
    """
    Plot 3D data recorded by BeamMonitor.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file that contains the data.
    dataset : {"current","emit","mean","std","cs_invariant"}
        HDF5 file's dataset to be plotted. The default is "mean".
    dimension : str
         The dimension of the dataset to plot:
            for "emit", dimension = {"x","y","s"},
            for "cs_invariant", dimension = {"x","y"},
            for "mean" and "std", dimension = {"x","xp","y","yp","tau","delta"}.
            not used if "current".
        The default is "tau".
    cm_lim : list [vmin, vmax], optional
        Colormap scale for the "streak" plot.

    Return
    ------
    fig : Figure
        Figure object with the plot on it.

    """
    
    file = hp.File(filename, "r")
    data = file["Beam"]
    time = np.array(data["time"])
        
    h = len(data["mean"][0,:,0])
    x = np.arange(h)
    x_label = "Bunch index"
    y = time
    y_label = "Number of turns"
    if dataset == "current":
        z = (np.array(data["current"])*1e3).T
        z_label = "Bunch current (mA)"
        title = z_label
    elif dataset == "emit":
        dimension_dict = {"x":0, "y":1, "s":2}
        axis = dimension_dict[dimension]
        label = ["$\\epsilon_{x}$ (m.rad)",
                 "$\\epsilon_{y}$ (m.rad)",
                 "$\\epsilon_{s}$ (s)"]
        z = np.array(data["emit"][axis,:,:]).T
        z_label = label[axis]
        title = z_label
    elif dataset == "cs_invariant":
        dimension_dict = {"x":0, "y":1}
        axis = dimension_dict[dimension]
        label = ['$J_x$ (m)', '$J_y$ (m)']
        z = np.array(data["cs_invariant"][axis,:,:]).T
        z_label = label[axis]
        title = z_label
    else:
        dimension_dict = {"x":0, "xp":1, "y":2, "yp":3, "tau":4, 
                              "delta":5}
        axis = dimension_dict[dimension]
        scale = [1e6, 1e6, 1e6, 1e6, 1e12, 1]
        label = ["x (um)", "x' ($\\mu$rad)", "y (um)", 
                     "y' ($\\mu$rad)", "$\\tau$ (ps)", "$\\delta$"]
        z = np.array(data[dataset][axis,:,:]).T*scale[axis]
        z_label = label[axis]
        if dataset == "mean":
            title = label[axis] + " CM"
        elif dataset == "std":
            title = label[axis] + " RMS"
            
    fig, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    if dataset == "mean":
        cmap = mpl.cm.coolwarm # diverging
    else:
        cmap = mpl.cm.inferno # sequential
    
    c = ax.imshow(z, cmap=cmap, origin='lower' , aspect='auto',
            extent=[x.min(), x.max(), y.min(), y.max()])
    if cm_lim is not None:
        c.set_clim(vmin=cm_lim[0],vmax=cm_lim[1])
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(z_label)
    
    file.close()
        
    return fig
              
def plot_bunchdata(filenames, bunch_number, dataset, dimension="x", 
                   legend=None):
    """
    Plot data recorded by BunchMonitor.
    
    Parameters
    ----------
    filenames : str or list of str
        Names of the HDF5 files to be plotted.
    bunch_number : int or list of int
        Bunch to plot. This has to be identical to 'bunch_number' parameter in 
        'BunchMonitor' object.
    dataset : {"current", "emit", "mean", "std", "cs_invariant"}
        HDF5 file's dataset to be plotted.
    dimension : str, optional
        The dimension of the dataset to plot. Use "None" for "current",
        otherwise use the following : 
            for "emit", dimension = {"x","y","s"},
            for "cs_invariant", dimension = {"x","y"},
            for "mean" and "std", dimension = {"x","xp","y","yp","tau","delta"},
            for "action", dimension = {"x","y"}.
    legend : list of str, optional
        Legend to add for each file.
        
    Return
    ------
    fig : Figure
        Figure object with the plot on it.

    """
    
    if isinstance(filenames, str):
        filenames = [filenames]
        
    if isinstance(bunch_number, int):
        ll = []
        for i in range(len(filenames)):
            ll.append(bunch_number)
        bunch_number = ll
        
    fig, ax = plt.subplots()
    
    for i, filename in enumerate(filenames):
        file = hp.File(filename, "r")
        group = "BunchData_{0}".format(bunch_number[i])  # Data group of the HDF5 file
        
        if dataset == "current":
            y_var = file[group][dataset][:]*1e3
            label = "current (mA)"
            
        elif dataset == "emit":
            dimension_dict = {"x":0, "y":1, "s":2}
                             
            y_var = file[group][dataset][dimension_dict[dimension]]
            
            if dimension == "x": label = "hor. emittance (m.rad)"
            elif dimension == "y": label = "ver. emittance (m.rad)"
            elif dimension == "s": label = "long. emittance (s)"
            
            
        elif dataset == "mean" or dataset == "std":                        
            dimension_dict = {"x":0, "xp":1, "y":2, "yp":3, "tau":4, "delta":5} 
            scale = [1e6, 1e6, 1e6, 1e6, 1e12, 1]        
            axis_index = dimension_dict[dimension]
            
            y_var = file[group][dataset][axis_index]*scale[axis_index]
            if dataset == "mean":
                label_list = ["x ($\\mu$m)", "x' ($\\mu$rad)", "y ($\\mu$m)",
                              "y' ($\\mu$rad)", "$\\tau$ (ps)", "$\\delta$"]
            else:
                label_list = ["$\\sigma_x$ ($\\mu$m)", "$\\sigma_{x'}$ ($\\mu$rad)",
                              "$\\sigma_y$ ($\\mu$m)", "$\\sigma_{y'}$ ($\\mu$rad)", 
                              "$\\sigma_{\\tau}$ (ps)", "$\\sigma_{\\delta}$"]
            
            label = label_list[axis_index]
            
        elif dataset == "cs_invariant":
            dimension_dict = {"x":0, "y":1}
            axis_index = dimension_dict[dimension]
            y_var = file[group][dataset][axis_index]
            label_list = ['$J_x$ (m)', '$J_y$ (m)']
            label = label_list[axis_index]

        x_axis = file[group]["time"][:]
        xlabel = "Number of turns"
        
        ax.plot(x_axis, y_var)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        if legend is not None:
            plt.legend(legend)
            
        file.close()
        
    return fig
            
def plot_phasespacedata(filename, bunch_number, x_var, y_var, turn,
                        only_alive=True, plot_size=1, plot_kind='kde'):
    """
    Plot data recorded by PhaseSpaceMonitor.

    Parameters
    ----------
    filename : str
        Name of the HDF5 file that contains the data.
    bunch_number : int
        Bunch to plot. This has to be identical to 'bunch_number' parameter in 
        'PhaseSpaceMonitor' object.
    x_var, y_var : str {"x", "xp", "y", "yp", "tau", "delta"}
        If dataset is "particles", the variables to be plotted on the 
        horizontal and the vertical axes need to be specified.
    turn : int
        Turn at which the data will be extracted.
    only_alive : bool, optional
        When only_alive is True, only alive particles are plotted and dead 
        particles will be discarded.
    plot_size : [0,1], optional
        Number of macro-particles to plot relative to the total number 
        of macro-particles recorded. This option helps reduce processing time
        when the data is big.
    plot_kind : {'scatter', 'kde', 'hex', 'reg', 'resid'}, optional
        The plot style. The default is 'kde'. 
        
    Return
    ------
    fig : Figure
        Figure object with the plot on it.
    """
    
    file = hp.File(filename, "r")
    
    group = "PhaseSpaceData_{0}".format(bunch_number)
    dataset = "particles"

    option_dict = {"x":0, "xp":1, "y":2, "yp":3, "tau":4, "delta":5}
    scale = [1e3,1e3,1e3,1e3,1e12,1]
    label = ["x (mm)","x' (mrad)","y (mm)","y' (mrad)","$\\tau$ (ps)",
             "$\\delta$"]
    
    # find the index of "turn" in time array
    turn_index = np.where(file[group]["time"][:]==turn) 
    
    if len(turn_index[0]) == 0:
        raise ValueError("Turn {0} is not found. Enter turn from {1}.".
                         format(turn, file[group]["time"][:]))     
    
    path = file[group][dataset]
    mp_number = path[:,0,0].size

    if only_alive is True:
        data = np.array(file[group]["alive"])
        index = np.where(data[:,turn_index])[0]
    else:
        index = np.arange(mp_number)
        
    if plot_size == 1:
        samples = index
    elif plot_size < 1:
        samples_meta = random.sample(list(index), int(plot_size*mp_number))
        samples = sorted(samples_meta)
    else:
        raise ValueError("plot_size must be in range [0,1].")
            
    # format : sns.jointplot(x_axis, yaxis, kind)
    x_axis = path[samples,option_dict[x_var],turn_index[0][0]]
    y_axis = path[samples,option_dict[y_var],turn_index[0][0]]    
        
    fig = sns.jointplot(x_axis*scale[option_dict[x_var]], 
                        y_axis*scale[option_dict[y_var]], kind=plot_kind)
   
    plt.xlabel(label[option_dict[x_var]])
    plt.ylabel(label[option_dict[y_var]])
            
    file.close()
    return fig

def plot_profiledata(filename, bunch_number, dimension="tau", start=0,
                     stop=None, step=None, profile_plot=True, streak_plot=True):
    """
    Plot data recorded by ProfileMonitor

    Parameters
    ----------
    filename : str
        Name of the HDF5 file that contains the data.
    bunch_number : int
        Bunch to plot. This has to be identical to 'bunch_number' parameter in 
        'ProfileMonitor' object.
    dimension : str, optional
        Dimension to plot. The default is "tau"
    start : int, optional
        First turn to plot. The default is 0.
    stop : int, optional
        Last turn to plot. If None, the last turn of the record is selected.
    step : int, optional
        Plotting step. This has to be divisible by 'save_every' parameter in
        'ProfileMonitor' object, i.e. step % save_every == 0. If None, step is
        equivalent to save_every.
    profile_plot : bool, optional
        If Ture, bunch profile plot is plotted.
    streak_plot : bool, optional
        If True, strek plot is plotted.

    Returns
    -------
    fig : Figure
        Figure object with the plot on it.

    """
    
    file = hp.File(filename, "r")
    path = file['ProfileData_{0}'.format(bunch_number)]
    l_bound = np.array(path["{0}_bin".format(dimension)])
    data = np.array(path[dimension])
    time = np.array(path["time"])
    
    if stop is None:
        stop = time[-1]
    elif stop not in time:
        raise ValueError("stop not found. Choose from {0}"
                         .format(time[:]))
 
    if start not in time:
        raise ValueError("start not found. Choose from {0}"
                         .format(time[:]))
    
    save_every = time[1] - time[0]
    
    if step is None:
        step = save_every
    
    if step % save_every != 0:
        raise ValueError("step must be divisible by the recording step "
                         "which is {0}.".format(save_every))
    
    dimension_dict = {"x":0, "xp":1, "y":2, "yp":3, "tau":4, "delta":5}
    scale = [1e6, 1e6, 1e6, 1e6, 1e12, 1]
    label = ["x (um)", "x' ($\\mu$rad)", "y (um)", "y' ($\\mu$rad)",
             "$\\tau$ (ps)", "$\\delta$"]
    
    num = int((stop - start)/step)
    n_bin = len(data[:,0])
    
    start_index = np.where(time[:] == start)[0][0]

    x_var = np.zeros((num+1,n_bin))
    turn_index_array = np.zeros((num+1,), dtype=int)
    for i in range(num+1):
        turn_index = int(start_index + i * step / save_every)
        turn_index_array[i] = turn_index
        # construct an array of bin mids
        x_var[i,:] = l_bound[:,turn_index]
        
    if profile_plot is True:
        fig, ax = plt.subplots()
        for i in range(num+1):
            ax.plot(x_var[i]*scale[dimension_dict[dimension]],
                    data[:,turn_index_array[i]], 
                    label="turn {0}".format(time[turn_index_array[i]]))
        ax.set_xlabel(label[dimension_dict[dimension]])
        ax.set_ylabel("number of macro-particles")         
        ax.legend()
            
    if streak_plot is True:
        turn = np.reshape(time[turn_index_array], (num+1,1))
        y_var = np.ones((num+1,n_bin)) * turn
        z_var = np.transpose(data[:,turn_index_array])
        fig2, ax2 = plt.subplots()
        cmap = mpl.cm.inferno # sequential
        c = ax2.imshow(z_var, cmap=cmap, origin='lower' , aspect='auto',
                       extent=[x_var.min()*scale[dimension_dict[dimension]],
                               x_var.max()*scale[dimension_dict[dimension]],
                               y_var.min(),y_var.max()])
        ax2.set_xlabel(label[dimension_dict[dimension]])
        ax2.set_ylabel("Number of turns")
        cbar = fig2.colorbar(c, ax=ax2)
        cbar.set_label("Number of macro-particles") 

    file.close()
    if profile_plot is True and streak_plot is True:
        return fig, fig2
    elif profile_plot is True:
        return fig
    elif streak_plot is True:
        return fig2
    
def plot_wakedata(filename, bunch_number, wake_type="Wlong", start=0,
                     stop=None, step=None, profile_plot=False, streak_plot=True,
                     bunch_profile=False, dipole=False):
    """
    Plot data recorded by WakePotentialMonitor

    Parameters
    ----------
    filename : str
        Name of the HDF5 file that contains the data.
    bunch_number : int
        Bunch to plot. This has to be identical to 'bunch_number' parameter in 
        'WakePotentialMonitor' object.
    wake_type : str, optional
        Wake type to plot: "Wlong", "Wxdip", ... 
    start : int, optional
        First turn to plot. The default is 0.
    stop : int, optional
        Last turn to plot. If None, the last turn of the record is selected.
    step : int, optional
        Plotting step. This has to be divisible by 'save_every' parameter in
        'WakePotentialMonitor' object, i.e. step % save_every == 0. If None, 
        step is equivalent to save_every.
    profile_plot : bool, optional
        If Ture, wake potential profile plot is plotted.
    streak_plot : bool, optional
        If True, strek plot is plotted.
    bunch_profile : bool, optional.
        If True, the bunch profile is plotted.
    dipole : bool, optional
        If True, the dipole moment is plotted.

    Returns
    -------
    fig : Figure
        Figure object with the plot on it.

    """
    
    file = hp.File(filename, "r")
    path = file['WakePotentialData_{0}'.format(bunch_number)]
    time = np.array(path["time"])
    
    if stop is None:
        stop = time[-1]
    elif stop not in time:
        raise ValueError("stop not found. Choose from {0}"
                         .format(time[:]))
 
    if start not in time:
        raise ValueError("start not found. Choose from {0}"
                         .format(time[:]))
    
    save_every = time[1] -time[0]
    
    if step is None:
        step = save_every
    
    if step % save_every != 0:
        raise ValueError("step must be divisible by the recording step "
                         "which is {0}.".format(save_every))
    
    dimension_dict = {"Wlong":0, "Wxdip":1, "Wydip":2, "Wxquad":3, "Wyquad":4}
    scale = [1e-12, 1e-12, 1e-12, 1e-15, 1e-15]
    label = ["$W_p$ (V/pC)", "$W_{p,x}^D (V/pC)$", "$W_{p,y}^D (V/pC)$", "$W_{p,x}^Q (V/pC/mm)$",
             "$W_{p,y}^Q (V/pC/mm)$"]
    
    if bunch_profile == True:
        tau_name = "tau_" + wake_type
        wake_type = "profile_" + wake_type
        dimension_dict = {wake_type:0}
        scale = [1]
        label = ["$\\rho$ (a.u.)"]
        cmap = mpl.cm.inferno # sequential
    elif dipole == True:
        tau_name = "tau_" + wake_type
        wake_type = "dipole_" + wake_type
        dimension_dict = {wake_type:0}
        scale = [1]
        label = ["Dipole moment (m)"]
        cmap = mpl.cm.coolwarm # diverging
    else:
        tau_name = "tau_" + wake_type
        cmap = mpl.cm.coolwarm # diverging
        
    data = np.array(path[wake_type])
        
    num = int((stop - start)/step)
    n_bin = len(data[:,0])
    
    start_index = np.where(time[:] == start)[0][0]
    
    x_var = np.zeros((num+1,n_bin))
    turn_index_array = np.zeros((num+1,), dtype=int)
    for i in range(num+1):
        turn_index = int(start_index + i * step / save_every)
        turn_index_array[i] = turn_index
        # construct an array of bin mids
        x_var[i,:] = np.array(path[tau_name])[:,turn_index]
                
    if profile_plot is True:
        fig, ax = plt.subplots()
        for i in range(num+1):
            ax.plot(x_var[i]*1e12,
                    data[:,turn_index_array[i]]*scale[dimension_dict[wake_type]], 
                    label="turn {0}".format(time[turn_index_array[i]]))
        ax.set_xlabel("$\\tau$ (ps)")
        ax.set_ylabel(label[dimension_dict[wake_type]])         
        ax.legend()
            
    if streak_plot is True:
        turn = np.reshape(time[turn_index_array], (num+1,1))
        y_var = np.ones((num+1,n_bin)) * turn
        z_var = np.transpose(data[:,turn_index_array]*scale[dimension_dict[wake_type]])
        fig2, ax2 = plt.subplots()
        c = ax2.imshow(z_var, cmap=cmap, origin='lower' , aspect='auto',
                       extent=[x_var.min()*1e12,
                               x_var.max()*1e12,
                               y_var.min(),y_var.max()])
        ax2.set_xlabel("$\\tau$ (ps)")
        ax2.set_ylabel("Number of turns")
        cbar = fig2.colorbar(c, ax=ax2)
        cbar.set_label(label[dimension_dict[wake_type]]) 

    file.close()
    if profile_plot is True and streak_plot is True:
        return fig, fig2
    elif profile_plot is True:
        return fig
    elif streak_plot is True:
        return fig2
    
def plot_bunchspectrum(filenames, bunch_number, dataset="incoherent", dim="tau",
                       turns=None, fs=None, log_scale=True, legend=None,
                       norm=False):
    """
    Plot coherent and incoherent spectrum data.

    Parameters
    ----------
    filenames : str or list of str
        Names of the HDF5 files to be plotted.
    bunch_number : int or list of int
        Bunch to plot. This has to be identical to 'bunch_number' parameter in 
        'BunchSpectrumMonitor' object.
    dataset : {"mean_incoherent", "coherent", "incoherent"}
        HDF5 file's dataset to be plotted. 
        The default is "incoherent".
    dim :  {"x","y","tau"}, optional
        The dimension of the dataset to plot.
        The default is "tau".
    turns : array or None, optional
        Numbers of the turns to plot.
        If None, all turns are shown. 
        The default is None.
    fs : float or None, optional
        If not None, the frequency axis is noramlised by fs. 
        The default is None.
    log_scale : bool, optional
        If True, the spectrum plots are shown in y-log scale. 
        The default is True.
    legend : list of str, optional
        Legend to add for each file.
        The default is None.
    norm : bool, optional
        If True, normalise the data of each spectrum by its geometric mean.
        The default is False.

    Return
    ------
    fig : Figure

    """
    
    if isinstance(filenames, str):
        filenames = [filenames]
        
    if isinstance(bunch_number, int):
        ll = []
        for i in range(len(filenames)):
            ll.append(bunch_number)
        bunch_number = ll
        
    fig, ax = plt.subplots()
    
    for i, filename in enumerate(filenames):
        file = hp.File(filename, "r")
        group = file["BunchSpectrum_{0}".format(bunch_number[i])]
        
        time = np.array(group["time"])
        freq = np.array(group["freq"])
        dim_dict = {"x":0, "y":1, "tau":2}
        
        if dataset == "mean_incoherent":
            y_var = group["mean_incoherent"][dim_dict[dim],:]
            y_err = group["std_incoherent"][dim_dict[dim],:]
            ax.errorbar(time, y_var, y_err)
            xlabel = "Turn number"
            ylabel = "Mean incoherent frequency [Hz]"
        elif dataset == "incoherent" or dataset == "coherent":
            
            if turns is None:
                turn_index = np.where(time == time)[0]
            else:
                turn_index = []
                for turn in turns:
                    idx = np.where(time == turn)[0][0]
                    turn_index.append(idx)
                turn_index = np.array(turn_index)
                
            if fs is None:
                x_var = freq
                xlabel = "Frequency [Hz]"
            else:
                x_var = freq/fs
                xlabel = r"$f/f_{s}$"
                
            for idx in turn_index:
                y_var = group[dataset][dim_dict[dim],:,idx]
                if norm is True:
                    y_var = y_var/gmean(y_var)
                ax.plot(x_var, y_var)
                
            if log_scale is True:
                ax.set_yscale('log')
                
            ylabel = "FFT amplitude [a.u.]"
            if dataset == "incoherent":
                ax.set_title("Incoherent spectrum")
            elif dataset == "coherent":
                ax.set_title("Coherent spectrum")            
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend is not None:
            plt.legend(legend)
        file.close()
        
    return fig

def streak_bunchspectrum(filename, bunch_number, dataset="incoherent", 
                         dim="tau", fs=None, log_scale=True, fmin=None, 
                         fmax=None, turns=None, norm=False, ylim=None):
    """
    Plot 3D data recorded by the BunchSpectrumMonitor.

    Parameters
    ----------
    filenames : str
        Name of the HDF5 file to be plotted.
    bunch_number : int
        Bunch to plot. This has to be identical to 'bunch_number' parameter in 
        'BunchSpectrumMonitor' object.
    dataset : {"coherent", "incoherent"}
        HDF5 file's dataset to be plotted. 
        The default is "incoherent".
    dim :  {"x","y","tau"}, optional
        The dimension of the dataset to plot.
        The default is "tau".
    fs : float or None, optional
        If not None, the frequency axis is noramlised by fs. 
    log_scale : bool, optional
        If True, the spectrum plots are shown in y-log scale. 
        The default is True.
    fmin : float, optional
        If not None, the plot is limitted to values bigger than fmin.
    fmax : float, optional
        If not None, the plot is limitted to values smaller than fmax.
    turns : array, optional
        If not None, only the turn numbers in the turns array are plotted.
    norm : bool, optional
        If True, normalise the data of each spectrum by its geometric mean.
        The default is False.
    ylim : array, optional
        If not None, should be array like in the form [ymin, ymax] where ymin 
        and ymax are the minimum and maxmimum values used in the y axis.
        

    Returns
    -------
    fig : Figure

    """
    
    file = hp.File(filename, "r")
    group = file["BunchSpectrum_{0}".format(bunch_number)]
    
    time = np.array(group["time"])
    freq = np.array(group["freq"])
    dim_dict = {"x":0, "y":1, "tau":2}
    
    if turns is None:
        turn_index = np.where(time == time)[0]
        if ylim is None:
            tmin = time.min()
            tmax = time.max()
        else:
            tmin = ylim[0]
            tmax = ylim[1]
    else:
        tmin = turns.min()
        tmax = turns.max()
        turn_index = []
        for turn in turns:
            idx = np.where(time == turn)[0][0]
            turn_index.append(idx)
        turn_index = np.array(turn_index)
    
    data = group[dataset][dim_dict[dim], :, turn_index]
    
    if log_scale is True:
        option = mpl.colors.LogNorm()
    else:
        option = None
    
    if fs is None:
        x_var = freq
        xlabel = "Frequency [Hz]"
    else:
        x_var = freq/fs
        xlabel = r"$f/f_{s}$"
        
    if fmin is None:
        fmin = x_var.min()
    if fmax is None:
        fmax = x_var.max()
        
    ind = (x_var > fmin) & (x_var < fmax)
    x_var=x_var[ind]
    data = data[ind,:]
    
    if norm is True:
        data = data/gmean(data)
    
    if ylim is None:
        ylabel = "Turn number"
    else:
        ylabel = ""
    
    fig, ax = plt.subplots()
    if dataset == "incoherent":
        ax.set_title("Incoherent spectrum")
    elif dataset == "coherent":
        ax.set_title("Coherent spectrum")   
        
    cmap = mpl.cm.inferno # sequential
    c = ax.imshow(data.T, cmap=cmap, origin='lower' , aspect='auto',
                  extent=[x_var.min(), x_var.max(), tmin, tmax],
                  norm=option, interpolation="none")
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_ylabel("FFT amplitude [a.u.]", rotation=270)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return fig

def plot_beamspectrum(filenames, dim="tau", turns=None, f0=None, 
                      log_scale=True, legend=None, norm=False):
    """
    Plot coherent beam spectrum data.

    Parameters
    ----------
    filenames : str or list of str
        Names of the HDF5 files to be plotted.
    dim :  {"x","y","tau"}, optional
        The dimension of the dataset to plot.
        The default is "tau".
    turns : array or None, optional
        Numbers of the turns to plot.
        If None, all turns are shown. 
        The default is None.
    f0 : float or None, optional
        If not None, the frequency axis is noramlised by f0. 
        The default is None.
    log_scale : bool, optional
        If True, the spectrum plots are shown in y-log scale. 
        The default is True.
    legend : list of str, optional
        Legend to add for each file.
        The default is None.
    norm : bool, optional
        If True, normalise the data of each spectrum by its geometric mean.
        The default is False.

    Return
    ------
    fig : Figure

    """
    
    if isinstance(filenames, str):
        filenames = [filenames]
        
    fig, ax = plt.subplots()
    
    for i, filename in enumerate(filenames):
        file = hp.File(filename, "r")
        group = file["BeamSpectrum"]
        
        dataset = "coherent"
        time = np.array(group["time"])
        freq = np.array(group["freq"])
        dim_dict = {"x":0, "y":1, "tau":2}
            
        if turns is None:
            turn_index = np.where(time == time)[0]
        else:
            turn_index = []
            for turn in turns:
                idx = np.where(time == turn)[0][0]
                turn_index.append(idx)
            turn_index = np.array(turn_index)
            
        if f0 is None:
            x_var = freq
            xlabel = "Frequency [Hz]"
        else:
            x_var = freq/f0
            xlabel = r"$f/f_{0}$"
            
        for idx in turn_index:
            y_var = group[dataset][dim_dict[dim],:,idx]
            if norm is True:
                y_var = y_var/gmean(y_var)
            ax.plot(x_var, y_var)
            
        if log_scale is True:
            ax.set_yscale('log')
            
        ylabel = "FFT amplitude [a.u.]"
        ax.set_title("Beam coherent spectrum")            
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend is not None:
            plt.legend(legend)
        file.close()
        
    return fig

def streak_beamspectrum(filename, dim="tau", f0=None, log_scale=True, fmin=None, 
                         fmax=None, turns=None, norm=False, ylim=None):
    """
    Plot 3D data recorded by the BeamSpectrumMonitor.

    Parameters
    ----------
    filenames : str
        Name of the HDF5 file to be plotted.
    dim :  {"x","y","tau"}, optional
        The dimension of the dataset to plot.
        The default is "tau".
    f0 : float or None, optional
        If not None, the frequency axis is noramlised by f0. 
    log_scale : bool, optional
        If True, the spectrum plots are shown in y-log scale. 
        The default is True.
    fmin : float, optional
        If not None, the plot is limitted to values bigger than fmin.
    fmax : float, optional
        If not None, the plot is limitted to values smaller than fmax.
    turns : array, optional
        If not None, only the turn numbers in the turns array are plotted.
    norm : bool, optional
        If True, normalise the data of each spectrum by its geometric mean.
        The default is False.
    ylim : array, optional
        If not None, should be array like in the form [ymin, ymax] where ymin 
        and ymax are the minimum and maxmimum values used in the y axis.

    Returns
    -------
    fig : Figure

    """
    
    file = hp.File(filename, "r")
    group = file["BeamSpectrum"]
    dataset="coherent"
    time = np.array(group["time"])
    freq = np.array(group["freq"])
    dim_dict = {"x":0, "y":1, "tau":2}
    
    if turns is None:
        turn_index = np.where(time == time)[0]
        if ylim is None:
            tmin = time.min()
            tmax = time.max()
        else:
            tmin = ylim[0]
            tmax = ylim[1]
    else:
        tmin = turns.min()
        tmax = turns.max()
        turn_index = []
        for turn in turns:
            idx = np.where(time == turn)[0][0]
            turn_index.append(idx)
        turn_index = np.array(turn_index)
    
    data = group[dataset][dim_dict[dim], :, turn_index]
    
    if log_scale is True:
        option = mpl.colors.LogNorm()
    else:
        option = None
    
    if f0 is None:
        x_var = freq
        xlabel = "Frequency [Hz]"
    else:
        x_var = freq/f0
        xlabel = r"$f/f_{0}$"
        
    if fmin is None:
        fmin = x_var.min()
    if fmax is None:
        fmax = x_var.max()
        
    ind = (x_var > fmin) & (x_var < fmax)
    x_var=x_var[ind]
    data = data[ind,:]
    
    if norm is True:
        data = data/gmean(data)
        
    if ylim is None:
        ylabel = "Turn number"
    else:
        ylabel = ""
    
    fig, ax = plt.subplots()
    ax.set_title("Beam coherent spectrum")   
        
    cmap = mpl.cm.inferno # sequential
    c = ax.imshow(data.T, cmap=cmap, origin='lower' , aspect='auto',
                  extent=[x_var.min(), x_var.max(), tmin, tmax],
                  norm=option, interpolation="none")
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.set_ylabel("FFT amplitude [a.u.]", rotation=270)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    return fig

def plot_cavitydata(filename, cavity_name, phasor="cavity", 
                    plot_type="bunch", bunch_number=0, turn=None, cm_lim=None):
    """
    Plot data recorded by CavityMonitor.

    Parameters
    ----------
    filename : str 
        Name of the HDF5 file that contains the data.
    cavity_name : str
        Name of the CavityResonator object.
    phasor : str, optional
        Type of the phasor to plot. Can be "beam" or "cavity".
    plot_type : str, optional
        Type of plot:
            - "bunch" plots the phasor voltage and angle versus time for a 
            given bunch.
            - "turn" plots the phasor voltage and ange versus bunch index for
            a given turn.
            - "streak_volt" plots the phasor voltage versus bunch index and 
            time.
            - "streak_angle" plots the phasor angle versus bunch index and 
            time.
            - "detune" or "psi" plots the detuning or tuning angle versus time.
            - "power" plots the generator, cavity, beam and reflected power
            versus time.
    bunch_number : int, optional
        Bunch number to select. The default is 0.
    turn : int, optional
        Turn to plot. The default is None.
    cm_lim : list [vmin, vmax], optional
        Colormap scale for the "streak" plots.

    Returns
    -------
    fig : Figure
        Figure object with the plot on it.

    """
    
    file = hp.File(filename, "r")
    cavity_data = file[cavity_name]
    
    time = np.array(cavity_data["time"])
    
    ph = {"cavity":0, "beam":1}
    labels = ["Cavity", "Beam"]
    
    if plot_type == "bunch":
    
        data = [cavity_data["cavity_phasor_record"][bunch_number,:], 
                cavity_data["beam_phasor_record"][bunch_number,:]]

        ylabel1 = labels[ph[phasor]] + " voltage [MV]"
        ylabel2 = labels[ph[phasor]] + " phase [rad]"
        
        fig, ax = plt.subplots()
        twin = ax.twinx()
        p1, = ax.plot(time, np.abs(data[ph[phasor]])*1e-6, color="r",label=ylabel1)
        p2, = twin.plot(time, np.angle(data[ph[phasor]]), color="b", label=ylabel2)
        ax.set_xlabel("Turn number")
        ax.set_ylabel(ylabel1)
        twin.set_ylabel(ylabel2)
        
        plots = [p1, p2]
        ax.legend(handles=plots, loc="best")
        
        ax.yaxis.label.set_color("r")
        twin.yaxis.label.set_color("b")
        
    if plot_type == "turn":
        
        index = np.array(time) == turn
        if (index.size == 0):
            raise ValueError("Turn is not valid.")
        ph = {"cavity":0, "beam":1}
        data = [np.array(cavity_data["cavity_phasor_record"])[:,index], 
                np.array(cavity_data["beam_phasor_record"])[:,index]]
        labels = ["Cavity", "Beam"]
        
        h=len(data[0])
        x=np.arange(h)

        ylabel1 = labels[ph[phasor]] + " voltage [MV]"
        ylabel2 = labels[ph[phasor]] + " phase [rad]"
        
        fig, ax = plt.subplots()
        twin = ax.twinx()
        p1, = ax.plot(x, np.abs(data[ph[phasor]])*1e-6, color="r",label=ylabel1)
        p2, = twin.plot(x, np.angle(data[ph[phasor]]), color="b", label=ylabel2)
        ax.set_xlabel("Bunch index")
        ax.set_ylabel(ylabel1)
        twin.set_ylabel(ylabel2)
        
        plots = [p1, p2]
        ax.legend(handles=plots, loc="best")
        
        ax.yaxis.label.set_color("r")
        twin.yaxis.label.set_color("b")
        
    if plot_type == "streak_volt" or plot_type == "streak_phase":
        
        if plot_type == "streak_volt":
            data = np.transpose(np.abs(cavity_data["cavity_phasor_record"][:,:])*1e-6)
            ylabel = labels[ph[phasor]] + " voltage [MV]"
            cmap = mpl.cm.coolwarm # diverging
        elif plot_type == "streak_phase":
            data = np.transpose(np.angle(cavity_data["cavity_phasor_record"][:,:]))
            ylabel = labels[ph[phasor]] + " phase [rad]"
            cmap = mpl.cm.coolwarm # diverging
            
        fig, ax = plt.subplots()
        c = ax.imshow(data, cmap=cmap, origin='lower' , aspect='auto')
        if cm_lim is not None:
            c.set_clim(vmin=cm_lim[0],vmax=cm_lim[1])
        ax.set_xlabel("Bunch index")
        ax.set_ylabel("Number of turns")
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label(ylabel)
        
    if plot_type == "detune" or plot_type == "psi":
        
        fig, ax = plt.subplots()
        if plot_type == "detune":
            data = np.array(cavity_data["detune"])*1e-3
            ylabel = r"Detuning $\Delta f$ [kHz]"
        elif plot_type == "psi":
            data = np.array(cavity_data["psi"])
            ylabel = r"Tuning angle $\psi$"
            
        ax.plot(time, data)
        ax.set_xlabel("Number of turns")
        ax.set_ylabel(ylabel)
        
    if plot_type == "power":
        Vc = np.mean(np.abs(cavity_data["cavity_phasor_record"]),0)
        theta = np.mean(np.angle(cavity_data["cavity_phasor_record"]),0)
        try:
            bunch_index = (file["Beam"]["current"][:,0] != 0).nonzero()[0]
            I0 = np.nansum(file["Beam"]["current"][bunch_index,:],0)
        except:
            print("Beam monitor is needed to compute power.")
            
        Rs = np.array(cavity_data["Rs"])
        Pc = Vc**2 / (2 * Rs)
        Pb = I0 * Vc * np.cos(theta)
        Pg = np.array(cavity_data["Pg"])
        Pr = Pg - Pb - Pc
        
        fig, ax = plt.subplots()
        ax.plot(time, Pg*1e-3, label="Generator power $P_g$ [kW]")
        ax.plot(time, Pb*1e-3, label="Beam power $P_b$ [kW]")
        ax.plot(time, Pc*1e-3, label="Dissipated cavity power $P_c$ [kW]")
        ax.plot(time, Pr*1e-3, label="Reflected power $P_r$ [kW]")
        ax.set_xlabel("Number of turns")
        ax.set_ylabel("Power [kW]")
        plt.legend()
        
    file.close()
    return fig
