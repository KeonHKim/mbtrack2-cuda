# -*- coding: utf-8 -*-
"""
Module where function used to import impedance and wakes from other codes are
defined.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.constants import c
from mbtrack2.impedance.wakefield import Impedance, WakeFunction, WakeField

def read_CST(file, component_type='long', divide_by=None):
    """
    Read CST file format into an Impedance object.
    
    Parameters
    ----------
    file : str
        Path to the file to read.
    component_type : str, optional
        Type of the Impedance object.
    divide_by : float, optional
        Divide the impedance by a value. Mainly used to normalize transverse 
        impedance by displacement.
        
    Returns
    -------
    result : Impedance object
        Data from file.
    """
    df = pd.read_csv(file, comment="#", header = None, sep = "\t", 
                    names = ["Frequency","Real","Imaginary"])
    df["Frequency"] = df["Frequency"]*1e9 
    if divide_by is not None:
        df["Real"] = df["Real"]/divide_by
        df["Imaginary"] = df["Imaginary"]/divide_by
    if component_type == "long":
        df["Real"] = np.abs(df["Real"])
    df.set_index("Frequency", inplace = True)
    result = Impedance(variable = df.index,
                       function = df["Real"] + 1j*df["Imaginary"],
                       component_type=component_type)
    return result

def read_IW2D(file, file_type='Zlong'):
    """
    Read IW2D file format into an Impedance object or a WakeField object.
    
    Parameters
    ----------
    file : str
        Path to the file to read.
    file_type : str, optional
        Type of the Impedance or WakeField object.
        
    Returns
    -------
    result : Impedance or WakeField object
        Data from file.
    """
    if file_type[0] == "Z":
        df = pd.read_csv(file, delim_whitespace=True, header = None, 
                         names = ["Frequency","Real","Imaginary"], skiprows=1)
        df.set_index("Frequency", inplace = True)
        df = df[df["Real"].notna()]
        df = df[df["Imaginary"].notna()]
        result = Impedance(variable = df.index,
                           function = df["Real"] + 1j*df["Imaginary"],
                           component_type=file_type[1:])
    elif file_type[0] == "W":
        df = pd.read_csv(file, delim_whitespace=True, header = None, 
                         names = ["Distance","Wake"], skiprows=1)
        df["Time"] = df["Distance"] / c
        df.set_index("Time", inplace = True)
        if np.any(df.isna()):
            index = df.isna().values
            df = df.interpolate()
            print("Nan values have been interpolated to:")
            print(df[index])
        # if file_type == "Wlong":
        #     df["Wake"] = df["Wake"]*-1
        result = WakeFunction(variable = df.index,
                           function = df["Wake"],
                           component_type=file_type[1:])
    else:
        raise ValueError("file_type should begin by Z or W.")
    return result

def read_IW2D_folder(folder, suffix, select="WZ"):
    """
    Read IW2D results into a WakeField object.
    
    Parameters
    ----------
    file : str
        Path to the file to read.
    suffix : str
        End of the name of each files. For example, in "Zlong_test.dat" the
        suffix should be "_test.dat".
    select : str, optional
        Select which object to load. "W" for WakeFunction, "Z" for Impedance 
        and "WZ" or "ZW" for both.
        
    Returns
    -------
    result : WakeField object
        WakeField object with Impedance and WakeFunction objects from the 
        different files.
    """
    if (select == "WZ") or (select == "ZW"):
        types = {"W" : WakeFunction,
                 "Z" : Impedance}
    elif (select == "W"):
        types = {"W" : WakeFunction}
    elif (select == "Z"):
        types = {"Z" : Impedance}
    else:
        raise ValueError("select should be W, Z or WZ.")
        
    components = ["long", "xdip", "ydip", "xquad", "yquad"]
    
    data_folder = Path(folder)
    
    list_for_wakefield = []
    for key, item in types.items():
        for component in components:
            name = data_folder / (key + component + suffix)
            res = read_IW2D(file=name, file_type=key + component)
            list_for_wakefield.append(res)
            
    wake = WakeField(list_for_wakefield)
    
    return wake