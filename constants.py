#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:21:32 2023

@author: kierapond
"""
import numpy as np
from scipy.special import gamma, factorial

# constants 
c = 299792458 # m (s^-1) -- speed of light 
h = 6.62607015e-34 # J s
hbar = h/(2*np.pi) # J s -- Planck's constant

mu0 = 4 * np.pi *1.00000000082e-7 # N A^-2 -no longer an exact value uncertainty of (20)-
epsilon0 = 1/(mu0 * c**2) #F (m^-1) -- dielectric constant
e = 1.602176634e-19 # C now exact
uatom = 1.66053906660e-27 # (50) kg -- atomic mass unit  codata 2018
kB = 1.380649e-23 #J (K^-1) -- Boltzman's constant new exact value 
me = 9.1093837015e-31 #kg (28) electron mass 2018 codata
a0 = 5.29177210903e-11 #m -bohr radius (80) codata 2018
muB = 9.274009994e-24 # J/T (57)-- Bohr magneton codata 2018
gS = 2.00231930436256 # Electron g-factor (35)

# 40 K 
LambdaD1 = 770.108136507e-9 #m
LambdaD2 = 766.700674872e-9 #m
NuD1 = c/LambdaD1 
kD1 = 2 *np.pi /LambdaD1 
NuD2 = c/LambdaD2
kD2 = 2 *np.pi/LambdaD2
CapitalGammaD1 = 2 *np.pi* 6.035e6 #Hz
CapitalGammaD2 = 2 *np.pi *6.035e6 #Hz
mK = 39.96399848 *uatom #error (21)

ahf = -h *285.7308e6 #For groundstate
gI = 0.000176490 #total nuclear g-factor 
gJ = 2.00229421 # For groundstate measured value 
gJ = gS #For theoretical value 

# def FeshbachField(states):
# 	Switch[Sort[states], {-9/2, -7/2}, 202.10, {-9/2, -5/2}, 
#    224.21, {-7/2, -7/2}, 198.8, _, 0];
# def FeshbachWidth(states):
# 	Switch[Sort[states], {-9/2, -7/2}, 7.8, {-9/2, -5/2}, 
#    9.7, {-7/2, -7/2}, 0, _, 0];
# def BackgroundScatLength(states): 
# 	Switch[Sort[states], {-9/2, -7/2}, 174 a0, {-9/2, -5/2}, 
#    0, {-7/2, -7/2}, 0, _, 0];

#Leaving this here as a historical artifact. Now we tend to define \
#a97 in terms of a97zero which takes the field value of the zero crossing as an argument. 
abg=167*a0
au = 1
C6 = 3897 *au
r0 = 1/np.sqrt(8) * gamma(3/4)/gamma(5/4) * ((mK *C6)/hbar**2)**(1/4)
r0 = 60*a0
CapitalDeltaB97 = 6.9
CapitalDeltB95 = 7.2
B097 = 202.10
B097zero = 209.115
B095 = 224.2
