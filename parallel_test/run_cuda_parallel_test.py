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
from mbtrack2_cuda.utilities import yokoya_elliptic
from mbtrack2_cuda.utilities import Optics
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

warnings.filterwarnings("ignore")

# Configuration
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
U0 = 1.01e6 # Energy loss per turn in [eV].
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

mp_number = int(10e4)
current = 400e-3 # [A]
gap = int(0)
m1 = 1
m2 = 3
Vc1 = 2.7e6 # [V]
Vc2 = np.sqrt(1/m2**2-(ring.U0/Vc1)**2*(1/(m2**2-1))) * Vc1 # [V]
theta1 = np.arccos(ring.U0/Vc1) # np.arccos(m2**2/(m2**2-1)*ring.U0/Vc1) # np.arccos(ring.U0/Vc1)
theta2 = -1*np.arccos((1-m2**2/(m2**2-1))*ring.U0/Vc2) # (0.01)*np.pi/180 - 0.5*np.pi # () corresponds to degree

turns_lrrw = int(50) # Number of turns to consider for the long range wakes

# 80 is enough for 100,000 macro-particles.
# 50 is enough for 50,000 macro-particles.
num_bin_gpu = int(80)
num_bin_cpu = num_bin_gpu

# After the binning the macro-particles within a bunch, we need to increase the # of bins for self-bunch RW calculations.
# 1000 is enough for most cases.
num_bin_gpu_interp = int(1000)

# Geometry and beam
long = LongitudinalMap(ring)
trans = TransverseMap(ring)
sr = SynchrotronRadiation(ring)
MCm = RFCavity(ring, m=m1, Vc=Vc1, theta=theta1)
MCh = RFCavity(ring, m=m2, Vc=Vc2, theta=theta2)

mybeam = Beam(ring)

# The zeroth index must be filled.
filling_pattern = np.zeros(ring.h)
n_fill = h - gap
step = 1
idx = 0
for _ in range(n_fill):
    filling_pattern[idx] = current / n_fill
    idx += step

mybeam.init_beam(filling_pattern, mp_per_bunch=mp_number, mpi=False, cuda=CUDA_PARALLEL)

#Resistive Wall: Resistive wall impedance

rho_Cu = 1.68e-8 # Copper's resistivity in (Ohm.m)
rho_Al = 3.32e-8 # A6063
rho_SS = 6.9e-7
Z0 = mu_0*c
radius_x = 9.5e-3
radius_y = 8e-3
length = L

wake_length = 100e-12
freq_lim = int(100e9)
freq_num = int(1e5)
freq = np.linspace(1, freq_lim, freq_num)
t = np.linspace(0, wake_length, freq_num)

## New
radius_y_al = 8e-3
radius_x_al = 9.5e-3
radius_y_ss = radius_y_al
radius_x_ss = radius_x_al
radius_y_ivu = 2.5e-3
radius_x_ivu = 7.5e-3
radius_y_epu = 7.5e-3 # circular
length_ivu = 60
length_epu = 15.208
length_ss = 100
length_al = L #- length_ivu - length_epu - length_ss

rw_al = CircularResistiveWall(t, freq, length=length_al, rho=rho_Al, radius=radius_y_al, exact=True)
rw_ss = CircularResistiveWall(t, freq, length=length_ss, rho=rho_SS, radius=radius_y_ss, exact=True)
rw_ivu = CircularResistiveWall(t, freq, length=length_ivu, rho=rho_Cu, radius=radius_y_ivu, exact=True)
rw_epu = CircularResistiveWall(t, freq, length=length_epu, rho=rho_Cu, radius=radius_y_epu, exact=True)

Wlong_al = rw_al.Wlong.data["real"].to_numpy()
Wxdip_al = rw_al.Wxdip.data["real"].to_numpy()
Wydip_al = rw_al.Wydip.data["real"].to_numpy()
Wlong_ss = rw_ss.Wlong.data["real"].to_numpy()
Wxdip_ss = rw_ss.Wxdip.data["real"].to_numpy()
Wydip_ss = rw_ss.Wydip.data["real"].to_numpy()
Wlong_ivu = rw_ivu.Wlong.data["real"].to_numpy()
Wxdip_ivu = rw_ivu.Wxdip.data["real"].to_numpy()
Wydip_ivu = rw_ivu.Wydip.data["real"].to_numpy()
Wlong_epu = rw_epu.Wlong.data["real"].to_numpy()
Wxdip_epu = rw_epu.Wxdip.data["real"].to_numpy()
Wydip_epu = rw_epu.Wydip.data["real"].to_numpy()

Y_al = yokoya_elliptic(radius_x_al, radius_y_al)
Y_ss = yokoya_elliptic(radius_x_ss, radius_y_ss)
Y_ivu = yokoya_elliptic(radius_x_ivu, radius_y_ivu)

#Total!
rw_total_Wlong = Y_al[0]*Wlong_al + Y_ss[0]*Wlong_ss + Y_ivu[0]*Wlong_ivu + Wlong_epu
rw_total_Wxdip = Y_al[1]*Wxdip_al + Y_ss[1]*Wxdip_ss + Y_ivu[1]*Wxdip_ivu + Wxdip_epu
rw_total_Wydip = Y_al[2]*Wydip_al + Y_ss[2]*Wydip_ss + Y_ivu[2]*Wydip_ivu + Wydip_epu

rw_total_Wlong_integ = np.cumsum(rw_total_Wlong)
rw_total_Wxdip_integ = np.cumsum(rw_total_Wxdip)
rw_total_Wydip_integ = np.cumsum(rw_total_Wydip)

types=["Wlong","Wxdip","Wydip"] # Wake types to consider.
r_lrrw = L / (length_al/radius_y_al + np.sqrt(rho_SS/rho_Al)*length_ss/radius_y_ss + np.sqrt(rho_Cu/rho_Al)*length_ivu/radius_y_ivu
         + np.sqrt(rho_Cu/rho_Al)*length_epu/radius_y_epu)
x3_lrrw = ( (Y_al[1]*length_al/radius_y_al**3 + np.sqrt(rho_SS/rho_Al)*Y_ss[1]*length_ss/radius_y_ss**3
        + np.sqrt(rho_Cu/rho_Al)*Y_ivu[1]*length_ivu/radius_y_ivu**3
        + np.sqrt(rho_Cu/rho_Al)*length_epu/radius_y_epu**3) / L)**(-1/3) # Horizontal effective radius of the 3rd power in [m], as Eq.27 in [1]. The default is radius.
y3_lrrw = ( (Y_al[2]*length_al/radius_y_al**3 + np.sqrt(rho_SS/rho_Al)*Y_ss[2]*length_ss/radius_y_ss**3
        + np.sqrt(rho_Cu/rho_Al)*Y_ivu[2]*length_ivu/radius_y_ivu**3
        + np.sqrt(rho_Cu/rho_Al)*length_epu/radius_y_epu**3) / L)**(-1/3) # Vertical effective radius of the 3rd power in [m], as Eq.27 in [1]. The default is radius.

print(y3_lrrw )

# rw = CircularResistiveWall(t, freq, length=L, rho=rho_Al, radius=radius_y, exact=True)
# Wlong = rw.Wlong.data["real"].to_numpy()
# Wxdip = rw.Wxdip.data["real"].to_numpy()
# Wydip = rw.Wydip.data["real"].to_numpy()
# Wlong_integ = np.cumsum(Wlong)
# Wxdip_integ = np.cumsum(Wxdip)
# Wydip_integ = np.cumsum(Wydip)
# print(Wlong_integ)
# print(Wxdip_integ)
#GPU Calculations
cumap = CUDAMap(ring, filling_pattern=filling_pattern, Vc1=Vc1, Vc2=Vc2, m1=m1, m2=m2, theta1=theta1, theta2=theta2,
                num_bin=num_bin_gpu, num_bin_interp=num_bin_gpu_interp, rho=rho_Cu,
                radius_x=radius_x, radius_y=radius_y, length=length, wake_function_time=t,
                wake_function_integ_wl=rw_total_Wlong_integ, wake_function_integ_wtx=rw_total_Wxdip_integ,
                wake_function_integ_wty=rw_total_Wydip_integ, r_lrrw=r_lrrw, x3_lrrw=x3_lrrw,
                y3_lrrw=y3_lrrw)

# Run simulation
print('Turns: ' + str(turns))
# culm: longitudinal map, cutm: transverse map, cusr: synchrotron radiation,
# curfmc: RF main cavity, curfhc: RF harmonic cavity, culrrw: long-range resistive wall
# cuelliptic: True means we use elliptical beam pipes, whereas False means we use circular beam pipes.
# curm: reduced monitor (If True, it does not monitor every turn.)

# Turn interval for monitoring -> It can be ignored if curm is False.
curm_ti = int(10)

for i in tqdm(range(1), desc='GPU Processing'):
    cumap.track(mybeam, turns, turns_lrrw, curm_ti, gap, culm=True, cusr=False, cutm=True, curfmc=True, curfhc=False,
                cusrrw=False, culrrw=True, cuelliptic=False, curm=False, cugeneralwake=False)

print("All tracking has been done.")