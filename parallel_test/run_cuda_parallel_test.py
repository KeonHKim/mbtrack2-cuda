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
from scipy.fft import irfft, ihfft
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
from scipy.constants import mu_0, c, pi
from scipy.special import erfc, wofz

warnings.filterwarnings("ignore")

# Configuration
CUDA_PARALLEL = True

# Parameters
h = 416 # Harmonic number of the accelerator.
L = 354.7 # Ring circumference in [m].
E0 = 2.75e9 # Nominal (total) energy of the ring in [eV].
particle = Electron() # Particle considered.
ac = 9.12e-5 # Momentum compaction factor.
U0 = 515e3 # Energy loss per turn in [eV].
tau = np.array([7.3e-3, 13.1e-3, 11.7e-3]) # Horizontal, vertical and longitudinal damping times in [s].
tune = np.array([54.2, 18.3]) # Horizontal and vertical tunes.
emit = np.array([80e-12, 25e-12]) # Horizontal and vertical equilibrium emittance in [m.rad].
sigma_0 = 8.2e-12 # Natural bunch length in [s].
sigma_delta = 9e-4 # Equilibrium energy spread.
chro = [1.6, 1.6] # Horizontal and vertical (non-normalized) chromaticities.

local_beta = np.array([3.1, 3.2]) # beta function for one-turn map
local_alpha = np.array([0.0, 0.0]) # alpha function for one-turn map
local_dispersion = np.array([0.0, 0.0, 0.0, 0.0]) # Dispersion function and its derivative for one-turn map

optics = Optics(local_beta=local_beta, local_alpha=local_alpha,
                  local_dispersion=local_dispersion)

ring = Synchrotron(h=h, optics=optics, particle=particle, L=L, E0=E0, ac=ac,
                   U0=U0, tau=tau, emit=emit, tune=tune,
                   sigma_delta=sigma_delta, sigma_0=sigma_0, chro=chro)

filling_pattern = np.zeros(ring.h) # We will define filling_pattern later.

mean_beta = np.array([4.24, 5.98])
norm_beta = mean_beta / local_beta

rho_Cu = 1.68e-8 # Copper's resistivity in [Ohm.m]
rho_Al = 3.32e-8 # A6063
rho = rho_Cu

Z0 = mu_0*c
radius_x = 6e-3 # x radius of beam pipe in [m] 
radius_y = 6e-3 # y radius of beam pipe in [m]
length = L # Total length of the beam pipes

freq_lim = 2000e13
num_t = int(8e4+1)
num_f = int(16e4)
idx_trunc = int(8e4)
freq = np.linspace(1, freq_lim, num_f)
tau_step = 1/(freq_lim)
tau_initial = np.arange(0, num_t) * tau_step
tau_array = tau_initial #tau_initial[:idx_trunc]
delta_tau = (tau_array[-1] - tau_array[0])/len(tau_array)
t0 = (2*rho*radius_y**2 / Z0)**(1/3) / c
kappa = 2*pi*t0*freq
kappa_interval = (kappa[-1] - kappa[0])/num_f
wl_circular = np.zeros_like(tau_array)
wt_circular = np.zeros_like(tau_array)

impl_circular = np.real(1/((1+1j)/np.sqrt(kappa)-1j*kappa/2))
impt_circular = np.real(1/((1+1j)*np.sqrt(kappa)-1j*kappa**2/2))
wl_circular_init = L * Z0*c/(2*pi**2*radius_y**2) * np.real(ihfft(impl_circular, norm="forward")) * 2 * kappa_interval
wt_circular_init = L * Z0*c**2*t0/(pi**2*radius_y**4) * np.imag(ihfft(impt_circular, norm="forward")) * 2 * kappa_interval
wl_circular_init[0] = 0 # Given that the integration from 0 to 0 must be zero, the first value of wl should be 0.
wl_circular = wl_circular_init #wl_circular_init[:idx_trunc]
wt_circular_init[0] = 0
wt_circular = wt_circular_init #wt_circular_init[:idx_trunc]

# Inside the CUDA kernels we consider the average values of wakes for each bin.
# Therefore, you need to perform cumulative summation before you put these wakes into CUDA kernels.
rw_total_Wlong_integ = np.cumsum(wl_circular) * delta_tau
rw_total_Wxdip_integ = np.cumsum(wt_circular) * delta_tau
rw_total_Wydip_integ = np.cumsum(wt_circular) * delta_tau

indices = [200/h] # Current array in [mA]
for idxs in tqdm(indices, desc="GPU"):
    turns = int(3e4)
    mp_number = int(10e4)
    current = idxs*1e-3 # current in [A]
    thread_per_block = int(16) # Choose int(16) or int(32). Select int(32) if the macro-particle count is int(2e6) or more.
    print(f"Bunch current: {current*1e3} [mA]")
    gap = int(0)
    m1 = 1
    m2 = 3
    Vc1 = 1.7e6 # Voltage of main cavity in [V]
    Vc2 = np.sqrt(1/m2**2-(ring.U0/Vc1)**2*(1/(m2**2-1))) * Vc1 # Voltage of harmonic cavity in [V]
    theta1 = np.arccos(ring.U0/Vc1) # np.arccos(m2**2/(m2**2-1)*ring.U0/Vc1) # np.arccos(ring.U0/Vc1)
    theta2 = -1*np.arccos((1-m2**2/(m2**2-1))*ring.U0/Vc2) # (0.01)*np.pi/180 - 0.5*np.pi # () corresponds to degree

    turns_lrrw = int(50) # Number of turns to consider for the long range wakes

    num_bin_gpu = int(100) # Number of bins for self-bunch wakes

    # num_bin_gpu_interp should be checked with longitudinal wake potential profile.
    # Interpolates the total number of bins for accurate calculations
    # Recommended to be less than int(1e4)
    num_bin_gpu_interp = int(6000)

    # GPU Calculations
    # Run simulation
    # culm: longitudinal map
    # cusr: synchrotron radiation & radiation damping
    # cutm: transverse map
    # curfmc: RF main cavity
    # curfhc: RF harmonic cavity
    # cusbwake: self-bunch wakes
    # culrrw: long-range (bunch-to-bunch) resistive wall

    # cuelliptic: True means we use elliptical beam pipes, whereas False means we use circular beam pipes.
    # curm: reduced monitor (If True, it does not monitor every turn.)

    # cugeneralwake: for self-bunch wake calculations, we calculate with arbitrary wakes that we want to use
    # cugeneralwakeshortlong: for self-bunch wake calculations, we use short-range resistive wall wakes for tau < t_crit
    # and we use long-range resistive wall wakes for tau > t_crit

    # monitordip; whether to save dipole moment data
    # dblPrec6D: whether to save 6D phase space coordinates as double-precision or single-precision


    # Important!
    # BunchToX=True, BunchToJ=False >> regular multi-bunch configuration
    # BunchToX=False, BunchToJ=False >> regular single bunch configuration
    # BunchToX=False, BunchToJ=True >> single bunch or multi-bunch configuration with number of macro-particle is greater than int(2e6)

    # cusbwake=False, cugeneralwake=False, cugeneralwakeshortlong=False >> We don't calculate any self-bunch wakes

    # cusbwake=True, cugeneralwake=False, cugeneralwakeshortlong=False,
    # curwcircularseries=True >> We calculate self-bunch resistive wall wakes with series expansion formulae

    # cusbwake=True, cugeneralwake=False, cugeneralwakeshortlong=False,
    # curwcircularseries=True >> We calculate self-bunch resistive wall wakes with Faddeeva function formulae

    # cusbwake=True, cugeneralwake=True, cugeneralwakeshortlong=False >> We calculate self-bunch arbitrary wakes

    # cusbwake=True, cugeneralwake=False, cugeneralwakeshortlong=True >> We calculate self-bunch resistive wall wakes but
    # the short-range wakes must be given by the user (This method is intended for the ellipitical or parallel plates case.)

    # cusbwake=True, cugeneralwake=True, cugeneralwakeshortlong=True >> This is not valid. you should use either one of
    # cugeneralwake=True or cugeneralwakeshortlong=True

    # rectangularaperture: consider the beam loss for rectangular aperture

    # Turn interval for monitoring -> It can be ignored if curm is False.
    curm_ti = int(10)
    num_bunch = 5
    for ind in range(num_bunch):
        filling_pattern[ind] = h * current / num_bunch

    mybeam = Beam(ring)
    mybeam.init_beam(filling_pattern, mp_per_bunch=mp_number, mpi=False, cuda=CUDA_PARALLEL)

    cumap = CUDAMap(ring, filling_pattern=filling_pattern, Vc1=Vc1, Vc2=Vc2, m1=m1, m2=m2, theta1=theta1, theta2=theta2,
                num_bin=num_bin_gpu, num_bin_interp=num_bin_gpu_interp, norm_beta=norm_beta, rho=rho,
                radius_x=radius_x, radius_y=radius_y, length=length, wake_function_time=tau_array,
                wake_function_integ_wl=rw_total_Wlong_integ, wake_function_integ_wtx=rw_total_Wxdip_integ,
                wake_function_integ_wty=rw_total_Wydip_integ, t_crit=20*t0, idxs=idxs,
                r_lrrw=0, x3_lrrw=0, y3_lrrw=0, x_aperture_radius=10, y_aperture_radius=10,
                thread_per_block=thread_per_block)

    cumap.track(mybeam, turns, turns_lrrw, curm_ti, culm=True, cusr=True, cutm=True, curfmc=True, curfhc=False,
            cusbwake=True, culrrw=True, cuelliptic=False, curm=False, cugeneralwake=False, cugeneralwakeshortlong=False,
            curwcircularseries=True, rectangularaperture=False, monitordip=False, dblPrec6D=False, BunchToX=True, BunchToJ=False)

print("All tracking has been done.")