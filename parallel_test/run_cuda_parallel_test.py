# Set the parent folder
import os
import sys

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Import packages
import warnings
import numpy as np
from scipy.constants import mu_0, c

from mbtrack2_cuda.impedance import CircularResistiveWall

from mbtrack2_cuda.tracking import Synchrotron, Electron
from mbtrack2_cuda.tracking import LongitudinalMap, TransverseMap, CUDAMap
from mbtrack2_cuda.tracking import SynchrotronRadiation
from mbtrack2_cuda.tracking import RFCavity
from mbtrack2_cuda.tracking import Beam, Bunch
from mbtrack2_cuda.tracking import WakePotential, LongRangeResistiveWall
from mbtrack2.utilities import yokoya_elliptic
from mbtrack2_cuda.utilities import Optics
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

warnings.filterwarnings("ignore")

# Configuration
# np.random.seed(1)
CUDA_PARALLEL = True
MPI_PARALLEL = False

MONITORING = False 
OUTPUT_FOLDER = "parallel_test"
OUTPUT_FILENAME = "tracking_gpu"

# Parameters
h = 1332 # Harmonic number of the accelerator.
L = 799.297 # Ring circumference in [m].
E0 = 4e9 # Nominal (total) energy of the ring in [eV].
particle = Electron() # Particle considered.
ac = 7.857e-5 # Momentum compaction factor.
U0 = 1.8e6 # Energy loss per turn in [eV].
tau = np.array([11.07e-3, 21.12e-3, 19.34e-3]) # Horizontal, vertical and longitudinal damping times in [s].
tune = np.array([68.18, 23.26]) # Horizontal and vertical tunes.
emit = np.array([62e-12, 6.2e-12]) # Horizontal and vertical equilibrium emittance in [m.rad].
sigma_0 = 12.21e-12 # Natural bunch length in [s].
sigma_delta = 1.26e-3 # Equilibrium energy spread.
chro = [5.8, 3.5] # Horizontal and vertical (non-normalized) chromaticities.

local_beta = np.array([8.42, 3.28]) # Beta function at the tracking location.
local_alpha = np.array([0.0, 0.0]) # Alpha function at the tracking location.
local_dispersion = np.array([0.0, 0.0, 0.0, 0.0]) # Dispersion function and its derivative at the tracking location.

optics = Optics(local_beta=local_beta, local_alpha=local_alpha, 
                  local_dispersion=local_dispersion)

ring = Synchrotron(h=h, optics=optics, particle=particle, L=L, E0=E0, ac=ac, 
                   U0=U0, tau=tau, emit=emit, tune=tune, 
                   sigma_delta=sigma_delta, sigma_0=sigma_0, chro=chro)

turns = int(3e4)

mp_number = int(4e4) # It should be at least greater than or equal to 175.
current = 3.8e-3
Vc1 = 3.5e6 # 2.7e6
Vc2 = 0.8e6
theta2 = (0.01)*np.pi/180 - 0.5*np.pi # () corresponds to degree
theta1 = np.arccos((ring.U0)/Vc1) # np.arccos((ring.U0-Vc2*np.cos(theta2))/Vc1)
m1 = 1
m2 = 3

turns_lrrw = int(22)

# 80 is enough for 100,000 macro-particles.
num_bin_gpu = int(50)
num_bin_cpu = num_bin_gpu

# 700 is enough for 100,000 macro-particles.
# 650 is enough for 50,000 macro-particles.
num_bin_gpu_interp = int(650)

# Geometry and beam
long = LongitudinalMap(ring)
trans = TransverseMap(ring)
sr = SynchrotronRadiation(ring)
MCm = RFCavity(ring, m=m1, Vc=Vc1, theta=theta1)
MCh = RFCavity(ring, m=m2, Vc=Vc2, theta=theta2)

mybeam = Beam(ring)

filling_pattern = np.zeros(ring.h)
n_fill = 100
step = 13
idx = 0
for _ in range(n_fill):
    filling_pattern[idx] = current
    idx += step

# filling_pattern = np.ones(ring.h) * current

mybeam.init_beam(filling_pattern, mp_per_bunch=mp_number, mpi=False, cuda=CUDA_PARALLEL)

#Resistive Wall: Resistive wall impedance

rho_Cu = 1.68e-8 # Copper's resistivity in (Ohm.m)
rho_Al = 3.32e-8 # A6063
# rho_Al_6061 = 3.99e-8
# # 2.82e-8 for regular Al
# rho_SS = 6.9e-7
Z0 = mu_0*c
radius_x = 23e-3 #10e-3
radius_y = 5e-3
length = L #40

#GPU Calculations
cumap = CUDAMap(ring, Vc1=Vc1, Vc2=Vc2, m1=m1, m2=m2, theta1=theta1, theta2=theta2,
                num_bin=num_bin_gpu, num_bin_interp=num_bin_gpu_interp, rho=rho_Al,
                radius_x=radius_x, radius_y=radius_y, length=length)

# Run simulation
print('Turns: ' + str(turns))
# culm: longitudinal map, cutm: transverse map, cusr: synchrotron radiation,
# curfmc: RF main cavity, curfhc: RF harmonic cavity, culrrw: long-range resistive wall
# curm: Reduce monitoring (If True, it does not monitor every turn.)

# Monitors for every curm_turns -> It can be ignored if curm is False.
curm_turns = int(10)

for i in tqdm(range(1), desc='GPU Processing'):
    cumap.track(mybeam, turns, turns_lrrw, curm_turns, culm=True, cusr=True, cutm=True, curfmc=True, curfhc=False, culrrw=False,
                curm=False)