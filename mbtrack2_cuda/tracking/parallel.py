# -*- coding: utf-8 -*-

"""
Module to handle parallel computation
"""

import numpy as np

class Mpi:
    """
    Class which handle parallel computation via the mpi4py module [1].
    
    Parameters
    ----------
    filling_pattern : bool array of shape (h,)
        Filling pattern of the beam, like Beam.filling_pattern
        
    Attributes
    ----------
    comm : MPI.Intracomm object
        MPI intra-comminicator of the processor group, used to manage 
        communication between processors. 
    rank : int
        Rank of the processor which run the program
    size : int
        Number of processor within the processor group (in fact in the 
        intra-comminicator group)
    table : int array of shape (size, 2)
        Table of correspondance between the rank of the processor and its 
        associated bunch number
    bunch_num : int
        Return the bunch number corresponding to the current processor
    next_bunch : int
        Return the rank of the next tracked bunch
    previous_bunch : int
        Return the rank of the previous tracked bunch
        
    Methods
    -------
    write_table(filling_pattern)
        Write a table with the rank and the corresponding bunch number for each
        bunch of the filling pattern
    rank_to_bunch(rank)
        Return the bunch number corresponding to rank
    bunch_to_rank(bunch_num)
        Return the rank corresponding to the bunch number bunch_num
    share_distributions(beam)
        Compute the bunch profiles and share it between the different bunches.
    share_means(beam)
        Compute the bunch means and share it between the different bunches.
        
    References
    ----------
    [1] L. Dalcin, P. Kler, R. Paz, and A. Cosimo, Parallel Distributed 
    Computing using Python, Advances in Water Resources, 34(9):1124-1139, 2011.
    """
    
    def __init__(self, filling_pattern):
        from mpi4py import MPI
        self.MPI = MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.write_table(filling_pattern)
        
    def write_table(self, filling_pattern):
        """
        Write a table with the rank and the corresponding bunch number for each
        bunch of the filling pattern
        
        Parameters
        ----------
        filling_pattern : bool array of shape (h,)
            Filling pattern of the beam, like Beam.filling_pattern
        """
        if(filling_pattern.sum() != self.size):
            raise ValueError("The number of processors must be equal to the"
                             "number of (non-empty) bunches.")
        table = np.zeros((self.size, 2), dtype = int)
        table[:,0] = np.arange(0, self.size)
        table[:,1] = np.where(filling_pattern)[0]
        self.table = table
    
    def rank_to_bunch(self, rank):
        """
        Return the bunch number corresponding to rank
        
        Parameters
        ----------
        rank : int
            Rank of a processor
            
        Returns
        -------
        bunch_num : int
            Bunch number corresponding to the input rank
        """
        return self.table[rank,1]
    
    def bunch_to_rank(self, bunch_num):
        """
        Return the rank corresponding to the bunch number bunch_num
        
        Parameters
        ----------
        bunch_num : int
            Bunch number
            
        Returns
        -------
        rank : int
            Rank of the processor which tracks the input bunch number
        """
        try:
            rank = np.where(self.table[:,1] == bunch_num)[0][0]
        except IndexError:
            print("The bunch " + str(bunch_num) + " is not tracked on any processor.")
            rank = None
        return rank
    
    @property
    def bunch_num(self):
        """Return the bunch number corresponding to the current processor"""
        return self.rank_to_bunch(self.rank)
    
    @property
    def next_bunch(self):
        """Return the rank of the next tracked bunch"""
        if self.rank + 1 in self.table[:,0]:
            return self.rank + 1
        else:
            return 0
    
    @property
    def previous_bunch(self):
        """Return the rank of the previous tracked bunch"""
        if self.rank - 1 in self.table[:,0]:
            return self.rank - 1
        else:
            return max(self.table[:,0])
        
    def share_distributions(self, beam, dimensions="tau", n_bin=75):
        """
        Compute the bunch profiles and share it between the different bunches.

        Parameters
        ----------
        beam : Beam object
        dimension : str or list of str, optional
            Dimensions in which the binning is done. The default is "tau".
        n_bin : int or list of int, optional
            Number of bins. The default is 75.

        """
        
        if(beam.mpi_switch == False):
            print("Error, mpi is not initialised.")
            
        if isinstance(dimensions, str):
            dimensions = [dimensions]
            
        if isinstance(n_bin, int):
            n_bin = np.ones((len(dimensions),), dtype=int)*n_bin
            
        bunch = beam[self.bunch_num]
        
        charge_per_mp_all = self.comm.allgather(bunch.charge_per_mp)
        self.charge_per_mp_all = charge_per_mp_all
            
        for i in range(len(dimensions)):
            
            dim = dimensions[i]
            n = n_bin[i]
            
            if len(bunch) != 0:
                bins, sorted_index, profile, center = bunch.binning(dimension=dim, n_bin=n)
            else:
                sorted_index = None
                profile = np.zeros((n-1,),dtype=np.int64)
                center = np.zeros((n-1,),dtype=np.float64)
                if beam.filling_pattern[self.bunch_num] is True:
                    beam.update_filling_pattern()
                    beam.update_distance_between_bunches()
               
            self.__setattr__(dim + "_center", np.empty((self.size, n-1), dtype=np.float64))
            self.comm.Allgather([center,  self.MPI.DOUBLE], [self.__getattribute__(dim + "_center"), self.MPI.DOUBLE])
            
            self.__setattr__(dim + "_profile", np.empty((self.size, n-1), dtype=np.int64))
            self.comm.Allgather([profile,  self.MPI.INT64_T], [self.__getattribute__(dim + "_profile"), self.MPI.INT64_T])
            
            self.__setattr__(dim + "_sorted_index", sorted_index)
            
    def share_means(self, beam):
        """
        Compute the bunch means and share it between the different bunches.

        Parameters
        ----------
        beam : Beam object

        """
        
        if(beam.mpi_switch == False):
            print("Error, mpi is not initialised.")
            
        bunch = beam[self.bunch_num]
        
        charge_all = self.comm.allgather(bunch.charge)
        self.charge_all = charge_all
        
        self.mean_all = np.empty((self.size, 6), dtype=np.float64)
        if len(bunch) != 0:
            mean = bunch.mean
        else:
            mean = np.zeros((6,), dtype=np.float64)
        self.comm.Allgather([mean, self.MPI.DOUBLE], [self.mean_all, self.MPI.DOUBLE])
                                