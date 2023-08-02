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
from mbtrack2_cuda.tracking import Beam
from mbtrack2_cuda.tracking import WakePotential
from mbtrack2_cuda.utilities import Optics
from tqdm import tqdm
import matplotlib.pyplot as plt

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
chro = [5.8, -3.5] # Horizontal and vertical (non-normalized) chromaticities.

local_beta = np.array([8.42, 3.28]) # Beta function at the tracking location.
local_alpha = np.array([0.0, 0.0]) # Alpha function at the tracking location.
local_dispersion = np.array([0.0, 0.0, 0.0, 0.0]) # Dispersion function and its derivative at the tracking location.

optics = Optics(local_beta=local_beta, local_alpha=local_alpha, 
                  local_dispersion=local_dispersion)

ring = Synchrotron(h=h, optics=optics, particle=particle, L=L, E0=E0, ac=ac, 
                   U0=U0, tau=tau, emit=emit, tune=tune, 
                   sigma_delta=sigma_delta, sigma_0=sigma_0, chro=chro)

mp_number = int(4e4) # It should be at least greater than or equal to 175.
Vc = 3.5e6

turns = int(3e4) #int(3e4)
turns_lrrw = int(20)
n_bin = int(100)

# Geometry and beam
long = LongitudinalMap(ring)
trans = TransverseMap(ring)
sr = SynchrotronRadiation(ring)
MC = RFCavity(ring, m=1, Vc=Vc, theta = np.arccos(ring.U0/Vc))

mybeam = Beam(ring)
filling_pattern = np.ones(ring.h) * 1e-3 # Bunches with 1 mA
mybeam.init_beam(filling_pattern, mp_per_bunch=mp_number, mpi=False, cuda=CUDA_PARALLEL)

#Resistive Wall: Resistive wall impedance
# wake_length = 100e-12 
# freq_lim = int(100e9)
# freq_num = int(200+1)
# freq = np.linspace(0, freq_lim, freq_num)
# t = np.linspace(-1*wake_length, wake_length, freq_num)

rho_Cu = 1.68e-8  # Copper's resistivity in (Ohm.m)
Z0 = mu_0*c
radius = 5e-3
length = 798.297 #40

# rw = CircularResistiveWall(t, freq, length=length, rho=rho_Cu, radius=radius, exact=False)

# rw_t = rw.Wlong.data.index
# rw_Wlong = rw.Wlong.data["real"]
# rw_Wxdip = rw.Wxdip.data["real"]
# rw_Wydip = rw.Wydip.data["real"]

# wake = WakePotential(ring, rw, n_bin=freq_num)

#GPU Calculations
cumap = CUDAMap(ring, m=1, Vc=Vc, theta=np.arccos(ring.U0/Vc), n_bin=n_bin, rho=rho_Cu, radius=radius, length=length)

# Run simulation
print('Turns: ' + str(turns))
# culm: longitudinal map, cutm: transverse map, cusr: synchrotron radiation,
# curfc: RF Cavity, curw: resistive wall, cubm: beam monitor

for i in tqdm(range(1), desc='GPU Processing'):
    cumap.track(mybeam, turns, turns_lrrw, culm=True, cutm=True, cusr=False, curfc=True, curw=False, cubm=True)

# print(mybeam.bunch_mean[4, 0])
# print(mybeam.bunch_mean[4, h-1])
# bunch_length_zeroth = mybeam.bunch_list[0]['tau']
# bunch_length_last = mybeam.bunch_list[h-1]['tau']
# print(bunch_length_zeroth.sum())
# print(bunch_length_last.sum())
print(mybeam.bunch_list[0])
print(mybeam.bunch_list[h-1])
print('cpu_beam_emitX (zeroth bunch): ' + str(mybeam.bunch_emit[0, 0]))
print('cpu_beam_emitX (last bunch): ' + str(mybeam.bunch_emit[0, h-1]))
print('cpu_beam_emitY (zeroth bunch): ' + str(mybeam.bunch_emit[1, 0]))
print('cpu_beam_emitY (last bunch): ' + str(mybeam.bunch_emit[1, h-1]))
print('cpu_beam_emitS (zeroth bunch): ' + str(mybeam.bunch_emit[2, 0]))
print('cpu_beam_emitS (last bunch): ' + str(mybeam.bunch_emit[2, h-1]))
# print(mybeam.bunch_particle)

# CPU test
# mybeam_cpu = Beam(ring)
# mybeam_cpu.init_beam(filling_pattern, mp_per_bunch=mp_number, mpi=MPI_PARALLEL, cuda=False)

# #Resistive Wall: Resistive wall impedance
# wake_length = 90e-12
# freq_lim = int(100e9)
# freq_num = int(1e5) #1e5
# freq = np.linspace(0, freq_lim, freq_num)
# t = np.linspace(0, wake_length, freq_num)

# rho_Cu = 1.68e-8  # Copper's resistivity in (Ohm.m)
# Z0 = mu_0*c
# radius = 5e-3

# rw = CircularResistiveWall(t, freq, length=L, rho=rho_Cu, radius=radius, exact=True)
# wake = WakePotential(ring, rw, n_bin=n_bin)

# for i in tqdm(range(turns), desc=f'Core #{mybeam.mpi.bunch_num if MPI_PARALLEL is True else 0} Processing'):

#     long.track(mybeam_cpu)
#     trans.track(mybeam_cpu)
#     # sr.track(mybeam_cpu)
#     MC.track(mybeam_cpu)
    
#     # Add the collective effects
#     wake.track(mybeam_cpu)

# # print(mybeam_cpu.bunch_list[0]['tau'].sum()/mp_number)
# # print(mybeam_cpu.bunch_list[h-1]['tau'].sum()/mp_number)
# print(mybeam_cpu.bunch_list[0])
# print(mybeam_cpu.bunch_list[h-1])
# print('cpu_beam_emitX (zeroth bunch): ' + str(mybeam_cpu.bunch_emit[0, 0]))
# print('cpu_beam_emitX (last bunch): ' + str(mybeam_cpu.bunch_emit[0, h-1]))
# print('cpu_beam_emitY (zeroth bunch): ' + str(mybeam_cpu.bunch_emit[1, 0]))
# print('cpu_beam_emitY (last bunch): ' + str(mybeam_cpu.bunch_emit[1, h-1]))
# print('cpu_beam_emitS (zeroth bunch): ' + str(mybeam_cpu.bunch_emit[2, 0]))
# print('cpu_beam_emitS (last bunch): ' + str(mybeam_cpu.bunch_emit[2, h-1]))

# # wake.plot_last_wake(wake_type="Wlong", plot_rho=True, plot_wake_function=True)
# # plt.show()

# # wake.plot_last_wake(wake_type="Wxdip", plot_rho=True, plot_dipole=True, plot_wake_function=True)
# # plt.show()

# # wake.plot_last_wake(wake_type="Wydip", plot_rho=True, plot_dipole=True, plot_wake_function=True)
# # plt.show()

# # Sanity check
# # np.testing.assert_allclose(mybeam.bunch_list[0]['x'], mybeam_cpu.bunch_list[0]['x'], atol=1e-8)
# # np.testing.assert_allclose(mybeam.bunch_list[0]['xp'], mybeam_cpu.bunch_list[0]['xp'], atol=1e-8)
# # np.testing.assert_allclose(mybeam.bunch_list[0]['y'], mybeam_cpu.bunch_list[0]['y'], atol=1e-9)
# # np.testing.assert_allclose(mybeam.bunch_list[0]['yp'], mybeam_cpu.bunch_list[0]['yp'], atol=1e-8)
# # np.testing.assert_allclose(mybeam.bunch_list[0]['tau'], mybeam_cpu.bunch_list[0]['tau'], atol=1e-17)
# # np.testing.assert_allclose(mybeam.bunch_list[0]['delta'], mybeam_cpu.bunch_list[0]['delta'], atol=1e-4)

# # np.testing.assert_allclose(mybeam.bunch_list[0]['x'], mybeam_cpu.bunch_list[0]['x'],
# #                            atol=1e-3*np.average(np.absolute(mybeam_cpu.bunch_list[0]['x'])))
# # np.testing.assert_allclose(mybeam.bunch_list[0]['xp'], mybeam_cpu.bunch_list[0]['xp'],
# #                            atol=1e-3*np.average(np.absolute(mybeam_cpu.bunch_list[0]['xp'])))
# # np.testing.assert_allclose(mybeam.bunch_list[0]['y'], mybeam_cpu.bunch_list[0]['y'],
# #                            atol=1e-3*np.average(np.absolute(mybeam_cpu.bunch_list[0]['y'])))
# # np.testing.assert_allclose(mybeam.bunch_list[0]['yp'], mybeam_cpu.bunch_list[0]['yp'],
# #                            atol=1e-3*np.average(np.absolute(mybeam_cpu.bunch_list[0]['yp'])))
# # np.testing.assert_allclose(mybeam.bunch_list[0]['tau'], mybeam_cpu.bunch_list[0]['tau'],
# #                            atol=1e-3*np.average(np.absolute(mybeam_cpu.bunch_list[0]['tau'])))
# # np.testing.assert_allclose(mybeam.bunch_list[0]['delta'], mybeam_cpu.bunch_list[0]['delta'],
# #                            atol=1e-3*np.average(np.absolute(mybeam_cpu.bunch_list[0]['delta'])))

print("All tracking has been done.")