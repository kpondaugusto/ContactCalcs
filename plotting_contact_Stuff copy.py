#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:46:29 2023

@author: kierapond
"""

from constants import * 
from scipy import optimize
from mpmath import *
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate
import scipy.special as special

B0 = 202.14 #G
Bzero = 209.07 #G
Ba = 0.1 #G 
Boff = B0 - 0.04 #G 
BvBEC = 201.98377695709598 
BcBCS = 202.28948343066662
BvUni = B0 + 0.01
abg = 167.6 * a0

omegabar = 2*np.pi*(170*440*440)**(1/3) #2 pi (trap freqx * trap freqy * trap freq z)**1/3
n = 34578 * 2 #atom number from morning tshots
EF3D = hbar * (3 * n)**(1/3) * omegabar #fermi energy 
ToTF = 0.53 #Temp from tshots with dimer pulse
kF = np.sqrt(2*mK*EF3D) / hbar 
T = ToTF * EF3D 
EFkHz = EF3D / h #kHz 
omega = 2* np.pi * 10000 

limit = 1000
timescale= 10**3

def Lambda(T):
	 return np.sqrt((2*np.pi*hbar**2)/(mK*kB*T))
 
def Lambda(ToTF):
	 return np.sqrt(4*np.pi)/(kF * np.sqrt(ToTF))
 
def polyfunc(z):
	return -polylog(3,-z)
 
def fugacity(ToTF):
	return optimize.root( polyfunc == 1/gamma(4)*ToTF**(-3), 1)

def a97thres(B):
	B = np.linspace(Boff-Ba, Boff+Ba, 100)
	a97 = abg * (1 - (Bzero - B0)/(B - B0))
	
	threshold = 100
	a97[a97*kF>threshold] = np.nan
	a97[a97*kF<-threshold] = np.nan
	
	return a97

def a97thres2(B):
	B = Bosc(tvalues/timescale)
	a97 = abg * (1 - (Bzero - B0)/(B - B0))
	
	threshold = 100
	a97[a97*kF>threshold] = np.nan
	a97[a97*kF<-threshold] = np.nan
	
	return a97

def a97(B):
	return abg * (1 - (Bzero - B0)/(B - B0))

def Bosc(t):
	return Ba * np.sin(omega*t) + Boff

def dia97(t, omega):
	return Ba/abg * ( ((Bzero - B0) *omega *np.cos(omega*t) ) / (Ba * np.sin(omega*t) + Boff - Bzero)**2 )

Bvalues = np.linspace(Boff-Ba, Boff+Ba, 100)

tvalues = np.linspace(0, 6*np.pi/(omega) * timescale, 100)

figure, axes = plt.subplots(2, 3) 


axes[0,0].set_xlabel('BField (G)')
axes[0,0].set_ylabel('kF a (dim)')
axes[0,0].plot(Bvalues, a97thres(Bvalues)*kF)

axes[0,1].set_xlabel('BField (G)')
axes[0,1].set_ylabel('1/kFa (dim)')
axes[0,1].plot(Bvalues, 1/a97(Bvalues))

axes[0,2].set_xlabel('Time (ms)')
axes[0,2].set_ylabel('B Field (G)')
axes[0,2].plot(tvalues, Bosc(tvalues/timescale))

axes[1,0].set_xlabel('Time (ms)')
axes[1,0].set_ylabel('kF a (dim)')
axes[1,0].plot(tvalues, kF*a97thres2(Bosc(tvalues/timescale)))

axes[1,1].set_xlabel('Time (ms)')
axes[1,1].set_ylabel('1/kFa (dim)')
axes[1,1].plot(tvalues, 1/a97(Bosc(tvalues/timescale))/kF)

axes[1,2].set_xlabel('Time (ms)')
axes[1,2].set_ylabel('d/domega 1/kFa (dim)')
axes[1,2].plot(tvalues, dia97(tvalues/timescale,omega)/kF/omega)


plt.figure(2)

plt.xlabel('Time (ms)')
plt.ylabel('B Field (G)')
# plt.plot(tvalues, 1/(kF*a97(Bosc(tvalues/timescale))))
plt.plot(tvalues, Bosc(tvalues/timescale))
plt.plot(tvalues, Bosc(tvalues/timescale+0.35/omega))

plt.show()

def CeqNorm(t, ToTF):
	return (1 + (Lambda(ToTF)/a97(Bosc(t)))/np.sqty(2)*(np.exp((Lambda(ToTF)/a97(Bosc(t)))**2 )/(2 *np.pi) ) * \
		 (1 + math.erf((Lambda(ToTF)/a97(Bosc(t))) /np.sqrt(2*np.pi)) ) )

def Ceq(t, ToTF):
	return 16*np.pi*fugacity(ToTF)**2*Lambda(ToTF)**(-4)* CeqNorm(t, ToTF)

def CeqPS(t, ToTF):
	return a97(Ba)*hbar*Ceq(t,ToTF)

def v(ToTF, B):
	return Lambda(ToTF)/(a97(B)*np.sqrt(2*np.pi))

def integrand(ToTF, omega, Boff, y):
	return (np.exp(-y)*np.sqrt(y*(y + ((hbar*omega)/(ToTF*EF3D)) ) )) / \
		((y +v(ToTF,Boff)**2)*(y + ((hbar*omega)/(ToTF*EF3D)) + v(ToTF,Boff)**2) )

def bulknorm(ToTF,omega,Boff,y):
	return ((1-np.exp((-hbar*omega)/(ToTF*EF3D)))/((hbar*omega)/(ToTF*EF3D))) * \
		(np.heaviside(v(ToTF,Boff)) *2 *v(ToTF,Boff)*np.exp(v(ToTF,Boff))**2 *\
	   (1/((hbar*omega)/(ToTF*EF3D)))*np.sqrt(((hbar*omega)/(ToTF*EF3D)) - v(ToTF,Boff)**2 )*\
		   np.heaviside((hbar*omega)/(ToTF*EF3D) - v(ToTF,Boff)**2) + \
		1/np.pi * integrate.quad(integrand, 0, limit, args=(ToTF,omega,Boff)) ) 

def bulk(ToTF,omega,Boff,y):
	return (2*np.sqrt(2))/9 * fugacity(ToTF)**2* Lambda(ToTF)**(-3)*v(ToTF,B)**2 * \
		bulknorm(ToTF,omega,Boff,y)

def bulkuni(ToTF,omega,Boff):
	return 2 *np.sqrt(2)/(9*np.pi) *fugacity(ToTF)**2*Lambda(ToTF)**(-3)*\
		v(ToTF,Boff)**2 *np.sinh((((hbar*omega))/(ToTF*EF3D))/2) / ((((hbar*omega))/(ToTF*EF3D))/2)*\
			kn(0, (((hbar*omega))/(ToTF*EF3D))/2 ) 

def bulkuninorm(ToTF,omega,Boff):
	return 1/np.pi * np.sinh((((hbar*omega))/(ToTF*EF3D))/2) / ((((hbar*omega))/(ToTF*EF3D))/2)*\
			kn(0, (((hbar*omega))/(ToTF*EF3D))/2) 
  
def Ctot(t, ToTF):
	return Ceq(t, ToTF) - 36*np.pi*mK*a97(Bosc(t))**2 * bulkuni(Totf,omega)*dia97(t)

omegavalues = np.linspace(0,1.5*kB*T/hbar,500)

plt.figure(2)

# plt.plot(omegavalues, bulkuni(ToTF,omegavalues,BvUni))

from numpy import loadtxt
bulkT58 = loadtxt("zetaomega_T0.58.txt", comments="#", delimiter=" ", unpack=False)

bulkT25 = loadtxt("zetaomega_T0.25.txt", comments="#", delimiter=" ", unpack=False)

bulkmeasshort = [0.00014,0.000131]
bulkshorterror = [0.00018,0.00011]
omegaoEFshort = [15/19,50/19]

bulkmeaslong = [0.000099,0.00013,0.00000567]
bulklongerror = [0.000042,0.0000025,0.0000004]
omegaoEF = [5/19,15/19,50/19]

bulkmeas202p1 = [0.00264,0.00121,0.00171,0.0364,0.00745,0.0022]
bulkerror202p1 = [0.00043,0.00053,0.00019,0.0038,0.0018,0.00019]
omegaoEF202p1 = [15/19,5/17,50/19,150/19,10/19,30/19]

bulkhot = []


fig = plt.figure(4)

ax = fig.add_subplot(111)

ax.set_xlabel('omega/EF')
ax.set_ylabel('bulk')
ax.loglog(bulkT58[:,0],bulkT58[:,1],label='T=.58')
ax.loglog(bulkT58[:,0],bulkT58[:,1],label='T=.58',marker='.')
ax.loglog(bulkT25[:,0],bulkT25[:,1],label='T=.25')
ax.loglog(bulkT25[:,0],bulkT25[:,1],label='T=.25',marker='d')
ax.errorbar(omegaoEF,bulkmeaslong,yerr = bulklongerror,marker='o',linestyle='None')
ax.errorbar(omegaoEF202p1,bulkmeas202p1,yerr = bulkerror202p1,marker='o',linestyle='None')

ax.set_yscale('log')
ax.show()

# fig.savefig('test.png')

