# -*- coding: utf-8 -*-
"""
Module where function used to import impedance and wakes from other codes are
defined.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile
from scipy.constants import c
from mbtrack2_cuda.impedance.wakefield import Impedance, WakeFunction, WakeField

def read_CST(file, component_type='long', divide_by=None, imp=True):
    """
    Read CST text file format into an Impedance or WakeFunction object.
    
    Parameters
    ----------
    file : str
        Path to the text file to read.
    component_type : str, optional
        Type of the Impedance or WakeFunction object to load.
        Default is 'long'.
    divide_by : float, optional
        Divide the impedance by a value. Mainly used to normalize transverse 
        impedance by displacement.
        Default is None.
    imp : bool, optional.
        If True a Impedance object is loaded, if False a WakeFunction object is
        loaded.
        Default is True.
        
    Returns
    -------
    result : Impedance or WakeFunction object.
        Data from file.
    """
    if imp:
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
    else:
        df = pd.read_csv(file, comment="#", header = None, sep = "\t", 
                        names = ["Distance","Wake"])
        df["Time"] = df["Distance"]*1e-3/c 
        df["Wake"] = df["Wake"]*1e12
        if divide_by is not None:
            df["Wake"] = df["Wake"]/divide_by
        df.set_index("Time", inplace = True)
        result = WakeFunction(variable = df.index,
                           function = df["Wake"],
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
    folder : str
        Path to the folder to read.
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

def read_ABCI(file, azimuthal=False):
    """
    Read ABCI output files [1].

    Parameters
    ----------
    file : str
        Path to ABCI .pot file.
    azimuthal : bool, optional
        If True, the transverse wake potential and impedance is loaded from the
        "AZIMUTHAL" data.
        If False, it is loaded from the "TRANSVERSE" data. In that case, a -1 
        factor is applied on the wake to agree with mbtrack2 sign convention.
        The default is False.

    Returns
    -------
    wake : WakeField
        Object where the ABCI computed impedance and wake are stored.
        
    References
    ----------
    [1] : ABCI - https://abci.kek.jp/abci.htm

    """
    
    if azimuthal:
        source="AZIMUTHAL"
    else:
        source="TRANSVERSE"
    
    def _read_temp(file, file_type, file2=None):
        if file_type[0] == "Z":
            df = pd.read_csv(file, delim_whitespace=True, 
                              names=["Frequency","Real"])
            df["Real"] = df["Real"]*1e3
            df["Frequency"] = df["Frequency"]*1e9
            df2 = pd.read_csv(file2, delim_whitespace=True, 
                              names=["Frequency","Imaginary"])
            df2["Imaginary"] = df2["Imaginary"]*1e3
            df2["Frequency"] = df2["Frequency"]*1e9
            df.set_index("Frequency", inplace = True)
            df2.set_index("Frequency", inplace = True)
            result = Impedance(variable = df.index,
                                function = df["Real"] + 1j*df2["Imaginary"],
                                component_type=file_type[1:])
        elif file_type[0] == "W":
            df = pd.read_csv(file, delim_whitespace=True, 
                              names=["Time","Wake"])
            df["Time"] = df["Time"] / c
            df["Wake"] = df["Wake"] * 1e12
            if (not azimuthal) and (file_type[-3:] == "dip"):
                df["Wake"] = df["Wake"] * -1
            df.set_index("Time", inplace = True)
            result = WakeFunction(variable = df.index,
                                function = df["Wake"],
                                component_type=file_type[1:])
        return result
    
    abci_dict = {'  TITLE: LONGITUDINAL WAKE POTENTIAL             \n':'Wlong',
                 '  TITLE: REAL PART OF LONGITUDINAL IMPEDANCE                      \n':'Zlong_re',
                 '  TITLE: IMAGINARY PART OF LONGITUDINAL IMPEDANCE                 \n':'Zlong_im',
                 f'  TITLE: {source} WAKE POTENTIAL               \n':'Wxdip',
                 f'  TITLE: REAL PART OF {source} IMPEDANCE                        \n':'Zxdip_re',
                 f'  TITLE: IMAGINARY PART OF {source} IMPEDANCE                   \n':'Zxdip_im'}
    wake_list = []
    start = True
    with open(file) as f:
        while True:
            if start is True:
                # read the header
                header = [next(f) for _ in range(5)]
                start = False
            elif line == '':
                # check if file is over
                break
            else:
                # read the header
                header = [next(f) for _ in range(4)]
                header.insert(0, line)
                
            # read the body until next TITLE field or end of file
            body = []
            while True:
                line = f.readline()
                try:
                    if line[:8] == "  TITLE:":
                        break
                    if line == '':
                        break
                except:
                    pass
                body.append(line)
            
            # write body in temp file and then process it into wake/imp
            try:
                if abci_dict[header[0]][0] == "W":
                    tmp = NamedTemporaryFile(delete=False, mode = "w+")
                    tmp.writelines(body)
                    tmp.flush()
                    tmp.close()
                    if abci_dict[header[0]][1:] == "long":
                        comp = _read_temp(tmp.name, abci_dict[header[0]])
                        wake_list.append(comp)
                    else:
                        comp_x = _read_temp(tmp.name, "Wxdip")
                        comp_y = _read_temp(tmp.name, "Wydip")
                        wake_list.append(comp_x)
                        wake_list.append(comp_y)
                    os.unlink(tmp.name)
                elif (abci_dict[header[0]][0] == "Z") and (abci_dict[header[0]][-2:] == "re"):
                    tmp1 = NamedTemporaryFile(delete=False, mode = "w+")
                    tmp1.writelines(body)
                    tmp1.flush()
                    tmp1.close()
                elif (abci_dict[header[0]][0] == "Z") and (abci_dict[header[0]][-2:] == "im"):
                    tmp2 = NamedTemporaryFile(delete=False, mode = "w+")
                    tmp2.writelines(body)
                    tmp2.flush()
                    tmp2.close()
                    if abci_dict[header[0]][1:-3] == "long":
                        comp = _read_temp(tmp1.name, "Zlong", tmp2.name)
                        wake_list.append(comp)
                    else:
                        comp_x = _read_temp(tmp1.name, "Zxdip", tmp2.name)
                        comp_y = _read_temp(tmp1.name, "Zydip", tmp2.name)
                        wake_list.append(comp_x)                        
                        wake_list.append(comp_y)                        
                    os.unlink(tmp1.name)
                    os.unlink(tmp2.name)
            except KeyError:
                pass
    
    wake = WakeField(wake_list)
    
    return wake

def read_ECHO2D(file, component_type='long'):
    """
    Read ECHO2D text file format (after matlab post-processing) into a 
    WakeFunction object.
    
    Parameters
    ----------
    file : str
        Path to the text file to read.
    component_type : str, optional
        Type of the WakeFunction object to load.
        Default is 'long'.
        
    Returns
    -------
    result : WakeFunction object.
        Data from file.
    """
    
    df = pd.read_csv(file, delim_whitespace=True, 
                     header = None, names = ["Distance","Wake"])
    df["Time"] = df["Distance"]/100/c
    df["Wake"] = df["Wake"]*1e12
    if component_type != 'long':
        df["Wake"] = df["Wake"]*-1
    df.set_index("Time", inplace = True)
    result = WakeFunction(variable = df.index,
                       function = df["Wake"],
                       component_type=component_type)
    
    return result
