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
from mbtrack2_cuda.tracking import LongitudinalMap, TransverseMap
from mbtrack2_cuda.tracking import SynchrotronRadiation
from mbtrack2_cuda.tracking import RFCavity
from mbtrack2_cuda.tracking import Beam
from mbtrack2_cuda.tracking import WakePotential

from mbtrack2_cuda.tracking.monitors import BeamMonitor

from mbtrack2_cuda.utilities import Optics

from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
CUDA_PARALLEL = False 

MONITORING = True
CUDA_MONITORING = CUDA_PARALLEL 
OUTPUT_FOLDER = "parallel_test"
OUTPUT_FILENAME = "tracking_gpu"

# Parameters
h = 4 # Harmonic number of the accelerator.
L = 799 # Ring circumference in [m].
E0 = 4e9 # Nominal (total) energy of the ring in [eV].
particle = Electron() # Particle considered.
ac = 7.857e-5 # Momentum compaction factor.
U0 = 1.8e6 # Energy loss per turn in [eV].
tau = np.array([11.07e-3, 21.12e-3, 19.34e-3]) # Horizontal, vertical and longitudinal damping times in [s].
tune = np.array([67.441, 23.168]) # Horizontal and vertical tunes.
emit = np.array([58e-12, 5.8e-12]) # Horizontal and vertical equilibrium emittance in [m.rad].
sigma_0 = 12.35e-12 # Natural bunch length in [s].
sigma_delta = 1.2e-3 # Equilibrium energy spread.
chro = [-115.1, -77.9] # Horizontal and vertical (non-normalized) chromaticities.

local_beta = np.array([8.42, 3.28]) # Beta function at the tracking location.
local_alpha = np.array([0, 0]) # Alpha function at the tracking location.
local_dispersion = np.array([0, 0, 0, 0]) # Dispersion function and its derivative at the tracking location.

optics = Optics(local_beta=local_beta, local_alpha=local_alpha, 
                  local_dispersion=local_dispersion)

ring = Synchrotron(h=h, optics=optics, particle=particle, L=L, E0=E0, ac=ac, 
                   U0=U0, tau=tau, emit=emit, tune=tune, 
                   sigma_delta=sigma_delta, sigma_0=sigma_0, chro=chro)

mp_number = 1e4
Vc = 3.5e6

turns = 2000

# Geometry and beam
long = LongitudinalMap(ring)
trans = TransverseMap(ring)
sr = SynchrotronRadiation(ring)
MC = RFCavity(ring, m=1, Vc=Vc, theta = np.arccos(ring.U0/Vc))

mybeam = Beam(ring)
filling_pattern = np.ones(ring.h) * 10e-3 # Bunches with 10 mA
mybeam.init_beam(filling_pattern, mp_per_bunch=mp_number, mpi=CUDA_PARALLEL)

#Resistive Wall: Resistive wall impedance
wake_length = 100e-12 
freq_lim = int(100e9)
freq_num = int(50)
freq = np.linspace(0, freq_lim, freq_num)
t = np.linspace(-1*wake_length, wake_length, freq_num)

rho_Cu = 1.68e-8  # Copper's resistivity in (Ohm.m)
Z0 = mu_0*c
radius = 5e-3

rw = CircularResistiveWall(t, freq, length=40, rho=rho_Cu, radius=radius, exact=False)
wake = WakePotential(ring, rw, n_bin=freq_num)

# Beam monitor is disabled temporarily

if MONITORING is True: 
    OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    beammonitor = BeamMonitor(h=ring.h, total_size=100, save_every=1, buffer_size=10, mpi_mode=CUDA_MONITORING, file_name=OUTPUT_PATH)
else:
    print('Beam monitor is disabled.')

# Run simulation
for i in tqdm(range(turns), desc='Processing'):

    long.track(mybeam)
# --------------------------------------- 여기서 부터 하나씩 쿠다 코드로 변경 --------------------------------------
    # trans.track(mybeam)
    # sr.track(mybeam)
    # MC.track(mybeam)
    
    # Add the collective effects
    # wake.track(mybeam)

    # Monitor
    if MONITORING is True:
        beammonitor.track(mybeam)

if MONITORING is True:
    beammonitor.close()

print("All tracking has been done.")