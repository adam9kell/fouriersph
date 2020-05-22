#### program fouriersph
# considers band bowing effects on valence band
# does not consider valence band degeneracy

import os
import numpy as np
from scipy.constants import pi, h, hbar, c, m_e, e, epsilon_0
from scipy.integrate import trapz
from scipy.linalg import solve, eigvalsh, eigh
from scipy.special import spherical_jn

file_path = os.path.abspath('fouriersph_v2.py')
index = [ind for ind, char in enumerate(file_path) if char == '\\']
file_path = file_path[:index[-1] + 1]

# vacuum level (/cm)
vac = 20000

# bulk band gap (/cm)
band_gap_core = 14030

# bulk effective masses (m0)
mass_elec_core = 0.11
mass_hole_core = 0.4

# static relative permittivity
stat_perm_core = 10.2

# bulk modulus (GPa)
bulk_mod_core = 53.3

# core bandgap volume dependence (/cm)
alpha_core = 18548

num_states = 5
num_fourier_terms = 25
num_comp_terms = 25
num = 160

ishell = input('what type of shell?  1=CdS, 2=ZnSe, 3=ZnS\n')

if ishell == '3':
    print('CdSe/ZnS')
    V_CB_shell = 8870
    V_VB_shell = 7260
    mass_elec_shell = 0.25
    mass_hole_shell = 1.3
    lattice_mismatch = 0.11
    bulk_mod_shell = 77.5
    alpha_shell = 23870
    
elif ishell == '2':
    print('CdSe/ZnSe')
    V_CB_shell = 5650
    V_VB_shell = 4500
    mass_elec_shell = 0.22
    mass_hole_shell = 0.6
    lattice_mismatch = 0.075
    bulk_mod_shell = 64.4
    alpha_shell = 40240
    
elif ishell == '1':
    print('CdSe/CdS')
    V_CB_shell = 380
    V_VB_shell = 6475
    mass_elec_shell = 0.21
    mass_hole_shell = 0.8
    lattice_mismatch = 0.044
    bulk_mod_shell = 63
    alpha_shell = 23870

d_core = float(input(' enter core diameter (nm)\n'))
h_shell = float(input(' enter shell thickness (nm)\n'))
Dt = float(input(' diffusion factor, Dt (0.01 = sharp interface)\n'))
compression_fraction = float(input('compression fraction (0.0 to 1.0)\n'))
compression_total = lattice_mismatch * compression_fraction

#core radius
r_core = d_core / 2
print((f'\n\ncore diameter{d_core:11.6f}'
       f'\nshell thickness{h_shell:11.6f}'
       f'\neffective lattice mismatch{compression_total:15.7e}'
       f'\ndiffusion parameter{Dt:11.7f}'))

r_total = r_core + h_shell
r_effective = r_total + 1.51
x1 = r_core / r_total
num2 = int(num * r_effective / r_total)

dx = 1 / num
dx2 = 1 / num2
dr = r_total / num

x = np.linspace(dx, 1, num = num, endpoint = True)
r1 = r_total * x
x2 = np.linspace(dx2, 1, num = num2, endpoint = True)
r2 =  r_effective * x2

# particle in a sphere eigenenergy (/cm)
pre_qce = hbar / 4 / pi / c / m_e / r_effective ** 2 * 1e16

# coulomb interaction energy (nm/cm)
pre_coul = e ** 2 / 4 / pi / h / c / epsilon_0 / stat_perm_core * 1e7 

# zeros of j0 and j1
z0 = np.arange(1, num_fourier_terms + 1) * pi
z1 = 1.36592 + 3.17771 * (z0 / pi) - 0.00218 * (z0 / pi) ** 2 + 0.000040105 * (z0 / pi) ** 3

#############################################
##### cation radial composition profile #####
#############################################

# initial composition step function
C0 = 1 - np.heaviside(x - x1, 0)

# radially-dependent composition profile
Ct = np.zeros(num)

if Dt <= 0.01:
    Ct = C0

else:
    # spherical bessel functions for superposition
    j_comp = spherical_jn(0, z1[:, None] * x)
    
    # normalization constants
    Aj = trapz(j_comp ** 2 * x ** 2, x = x)
    
    # superposition coefficients
    An = trapz(j_comp * C0 * x ** 2, x = x)
    
    # time-dependent diffusion
    diffusion = np.exp(-Dt * (z1 / r_total) ** 2)
    
    # composition profile after diffusion
    Ct = np.sum((An * diffusion / Aj)[:, None] * j_comp, axis = 0)

    # normalization
    ave0 = trapz(C0 * x ** 2, x = x)
    ave1 = trapz(Ct * x ** 2, x = x)
    avet = trapz(x ** 2, x = x)
    Ct += (ave0 - ave1) / avet

np.clip(Ct, 0, 1, Ct)

#############################
##### stress and strain #####
#############################

# calculated from elastic continuum theory
# saada, a. s. elastic theory and applications
# permagon press: new york, 1974

# poisson's ratio
poisson_ratio_core = 0.34
t11 = (1 - 2 * poisson_ratio_core) / (1 + poisson_ratio_core)

# radial displacement equation factors
young_modulus = 3 * (bulk_mod_core * Ct + bulk_mod_shell * (1 - Ct)) * (1 - 2 * poisson_ratio_core)
pre_disp = (1 + poisson_ratio_core) / young_modulus
rcp = (r1[1:] / r1[:-1]) ** 3
rcpi = 1 / rcp

Ai_plus = pre_disp[1:] * r1[1:] * (t11 + 0.5) / (rcp - 1)

Bi_plus = pre_disp[1:] * r1[1:] * (t11 + 0.5 * rcpi) / (rcpi - 1)
Bi_plus = np.insert(Bi_plus, 0, -pre_disp[0] * r1[0] * t11)

Ai_minus = pre_disp[1:] * r1[:-1] * (t11 + 0.5 * rcp) / (rcp - 1)

Bi_minus = pre_disp[1:-1] * r1[:-2] * (t11 + 0.5) / (rcpi[:-1] - 1)

disp_matrix = np.diag(Bi_plus[:-1] - Ai_minus) + np.diag(Ai_plus[:-1], k = -1) + np.diag(-Bi_minus, k = 1)

lattice_diff = r1[:-1] * compression_total * np.diff(Ct)

# radially-dependent pressure
P = solve(disp_matrix, lattice_diff)

# boundary condition sets P = 0 at particle surface
P = np.abs(np.append(P, 0))

# displacement
disp = P[:-1] * Ai_plus + P[1:] * Bi_plus[1:]

# tangential compnent of stress tensor
stress_tang = P[:-1] * 1.5 / (rcp - 1) - P[1:] * 0.5 * (rcpi + 2) / (1 - rcpi)

# radial and tangential strain
strain_rad = (np.diff(disp) + lattice_diff[1:]) / dr
strain_tang = disp / r1[1:]
strain_v = strain_rad + 2 * strain_tang[1:]
strain_v = np.insert(strain_v, 0, strain_v[0])
strain_v = np.append(strain_v, 0)

P_shift = -strain_v * (Ct * alpha_core + (1 - Ct) * alpha_shell)

# total strain energy and strain energy density
strain_energy_rad = -4 * pi * trapz(strain_rad * P[1:-1] * r1[1:-1] ** 2, x = r1[1:-1])
strain_energy_tang = 8 * pi * trapz(stress_tang[1:] * strain_tang[1:] * r1[1:-1] ** 2, x = r1[1:-1])
strain_energy_total = 6.242 * (strain_energy_rad + strain_energy_tang)
strain_energy_density = strain_energy_total / 4 / pi / r_core ** 2

print((f'core pressure{P[9]:11.6f}'
       f'\ntotal strain energy (eV){strain_energy_total:11.5f}'
       f'\nstrain energy density (eV/nm2){strain_energy_density:11.7f}'))

#########################
##### wavefunctions #####
#########################

# calculate normalized spherical bessel functions

# spherical bessel functions for superposition
j0 = spherical_jn(0, z0[:, None] * x2)

# normalization constants
A = trapz(j0 ** 2 * r2 ** 2, x = r2)

j0 /= np.sqrt(A)[:, None]

########################################################
##### matrix representation of s state hamiltonian #####
########################################################

band_gap_scale = band_gap_core + P_shift[9]

# radially-dpenedent hole effective mass
mass_hole = mass_hole_core + (1 - Ct) * (mass_hole_shell - mass_hole_core)
mass_hole = np.append(mass_hole, np.ones(num2 - num))

# radially-dependent potential energies
V_CB = (1 - Ct) * V_CB_shell + P_shift
V_CB = np.append(V_CB, vac * np.ones(num2 - num))

bow = 0
V_VB = (1 - Ct) * (V_VB_shell - bow * Ct)
V_VB = np.append(V_VB, vac * np.ones(num2 - num))

# calculate the kinetic and potential terms
# T1 is usually kinetic operator
T1_hole = trapz(j0[:, None] / mass_hole * j0 * r2 ** 2, x = r2)
T1_hole = pre_qce * z0 ** 2 * np.tril(T1_hole)

# T2 results from position-dependent mass
T2_hole = -trapz((j0[:, 1:] - j0[:, :-1])[:, None] * (1 / mass_hole[1:] - 1 / mass_hole[:-1]) * j0[:, :-1] * r2[:-1] ** 2, x = r2[:-1])
T2_hole = pre_qce * np.tril(T2_hole)

# V is radially-dependent potential term
V_elec = trapz(j0[:, None] * V_CB * j0 * r2 ** 2, x = r2)
V_elec = np.tril(V_elec)

V_hole = trapz(j0[:, None] * V_VB * j0 * r2 ** 2, x = r2)
V_hole = np.tril(V_hole)

####################################
##### zero-order wavefunctions #####
####################################

# hole eigenenergies and eigenvectors
e0_hole, eig_vec = eigh(T1_hole + T2_hole + V_hole, eigvals = (0, num_states - 1))
eig_vec = eig_vec.T

fh0 = np.sum(eig_vec[:, :, None] * j0, axis = 1)
tot = trapz(fh0 ** 2 * r2 ** 2, x = r2)
fh0 /= np.sqrt(tot)[:, None]

flag = 0
scale = 1

while flag <= 3:
    
    # radially-dependent electron effective mass
    mass_core_scale = scale * mass_elec_core
    mass_shell_scale = scale * mass_elec_shell

    mass_elec = mass_core_scale + (1 - Ct) * (mass_shell_scale - mass_core_scale)
    mass_elec = np.append(mass_elec, np.ones(num2 - num))

    # calculate the electron kinetic terms
    T1_elec = trapz(j0[:, None] / mass_elec * j0 * r2 ** 2, x = r2)
    T1_elec = pre_qce * z0 ** 2 * np.tril(T1_elec)

    T2_elec = trapz((j0[:, 1:] - j0[:, :-1])[:, None] * (1 / mass_elec[1:] - 1 / mass_elec[:-1]) * j0[:, :-1] * r2[:-1] ** 2, x = r2[:-1])
    T2_elec = -pre_qce * np.tril(T2_elec)

    # lowest electron eigenenergy
    e0_elec = eigvalsh(T1_elec + T2_elec + V_elec, eigvals = (0, 0))

    # empirical scaling of electron effective mass
    scale = 0.36773 + 2.7563E-4 * (e0_elec + e0_hole[0]) - 8.31053E-9 * (e0_elec + e0_hole[0]) ** 2

    flag += 1

# electron eigenenergies and eigenvectors
e0_elec, eig_vec = eigh(T1_elec + T2_elec + V_elec, eigvals = (0, num_states - 1))
eig_vec = eig_vec.T

# zero-order electron wavefunctions
fe0 = np.sum(eig_vec[:, :, None] * j0, axis = 1)
tot = trapz(fe0 ** 2 * r2 ** 2, x = r2)
fe0 /= np.sqrt(tot)[:, None]

#############################################
##### electron-hole coulomb interaction #####
#############################################

# r is array containing max(re, rh)
r = np.tile(r2, (num2, 1))
r = np.triu(r) + np.tril(r.T, -1)

# first-order wavefunction corrections
fe1 = pre_coul / stat_perm_core * trapz(fe0[0] * fe0[1:] * r2 ** 2 * trapz(fh0[0] ** 2 / r * r2 ** 2, x = r2), x = r2)
fe1 = (fe1 / (e0_elec - e0_elec[0])[1:])[:, None] * fe0[1:]

fh1 = pre_coul / stat_perm_core * trapz(fh0[0] * fh0[1:] * r2 ** 2 * trapz(fe0[0] ** 2 / r.T * r2 ** 2, x = r2), x = r2)
fh1 = (fh1 / (e0_hole - e0_hole[0])[1:])[:, None] * fe0[1:]

# total wavefunctions for lowest energy state
fe = fe0[0] + np.sum(fe1, axis = 0)
fh = fh0[0] + np.sum(fh1, axis = 0)

# renormalize wavefunctions
A_fe = trapz(fe ** 2 * r2 ** 2, x = r2)
A_fh = trapz(fh ** 2 * r2 ** 2, x = r2)

fe /= np.sqrt(A_fe)
fh /= np.sqrt(A_fh)

# calculate coulomb energy of first-order corrected functions
e_coul = pre_coul * trapz(fe ** 2 * r2 ** 2 * trapz(fh ** 2 / r * r2 ** 2, x = r2), x = r2)

# calculate the electron hole overlap
s = trapz(fh * fe * r2 ** 2, x = r2) ** 2

#############################
##### output parameters #####
#############################

# extend particle only parameters out to entire calculation range
C0 = np.append(C0, np.zeros(num2 - num))
Ct = np.append(Ct, np.zeros(num2 - num))
lattice_diff = np.append(lattice_diff, np.zeros(num2 - num + 1))
P = np.append(P, np.zeros(num2 - num))

with open(file_path + 'composition.txt', 'w') as composition:
    for i in range(num2):
        composition.write(f'{r2[i]:15.8f}{C0[i]:15.8f}{Ct[i]:15.8f}{V_CB[i]:15.8f}{V_VB[i]:15.8f}{lattice_diff[i]:15.8f}{P[i]:15.8f}\n')

with open(file_path + 'function.txt', 'w') as function:
    for i in range(num2):
        function.write(f'{r2[i]:15.8f}{fe[i]:15.8f}{fh[i]:15.8f}\n')

wavelength = 1e7 / (band_gap_scale + e0_elec[0] + e0_hole[0] - e_coul)

print((f'\nelectron quantum confinment energy{e0_elec[0]:11.3f}'
       f'\nhole quantum confinment energy{e0_hole[0]:11.3f}'
       f'\nelectron-hole interaction energy{e_coul:11.4f}'
       f'\nelectron-hole overlap ={s:11.7f}'
       f'\nonset wavelength {wavelength:11.4f}'))
