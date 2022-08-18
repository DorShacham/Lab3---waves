# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 08:44:28 2022

@author: student
"""

#%% Inductance
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit

def flux(voltage,time):
    flux = scipy.integrate.cumtrapz(voltage,x=time,initial=0)
    return (flux)

def arg_max_flux(flux,time):
    index = np.argmax(flux)
    return time[index]

def linearCurve(a,b):
    return a*x+b

h = np.array([30,24,18,14,8]) * 1e-2 #in meter
plt.figure()
Ind_data = []
for n in range(0,5):
    df = pd.read_csv("Trace %d.csv"%n, header = 1)
    Ind_data.append(df)
    t = df["Time (s)"]
    ref = df["1 (VOLT)"]
    signal = df["2 (VOLT)"]
    plt.plot(t,ref,".", label = "ref %d"%n)
    plt.plot(t,signal,".", label = "signal %d"%n)
    
    
plt.grid()
plt.legend(ncol = 5,loc='lower center', bbox_to_anchor=(0.5, -0.35))
plt.xlabel("Time [s]")
plt.ylabel("Voltage[V]")
plt.show()

t_coil =[]
for n in range(0,5):
    df = Ind_data[n]
    t = df["Time (s)"]
    ref = df["1 (VOLT)"]
    signal = df["2 (VOLT)"]
    ref_flux = flux(ref,t)
    sig_flux = flux(signal,t)
    
    plt.plot(t,ref_flux,".", label = "ref flux %d"%n)
    plt.plot(t,sig_flux,".", label = "signal flux %d"%n)
    
    ref_ind_max = np.argmax(ref_flux)
    sig_ind_max = np.argmax(sig_flux)
    plt.plot(t[ref_ind_max],ref_flux[ref_ind_max],"ro")
    plt.plot(t[sig_ind_max],sig_flux[sig_ind_max],"ro")
    
    t_coil.append(t[sig_ind_max]-t[ref_ind_max])
    
    


plt.grid()
plt.legend(ncol = 5,loc='lower center', bbox_to_anchor=(0.5, -0.35))
plt.xlabel("Time [s]")
plt.ylabel("Flux")
plt.show()

t_coil = np.array(t_coil)
t_coil_err = 1e-3
h_err = 1e-3
h_over_t_error = h/t_coil*np.sqrt((t_coil_err/t_coil)**2+(h_err/h)**2)
x = t_coil
y =h/t_coil
plt.figure()
plt.errorbar(x,y,yerr=h_over_t_error,xerr=t_coil_err,fmt = ".",label="data")
plt.grid()
plt.xlabel("t_coil [s]")
plt.ylabel("h/_tcoil [m/s]")

reg = linregress(x,y)
plt.plot(x,linearCurve(reg.slope, reg.intercept),label="regression")

plt.legend()
plt.show()


v0 = reg.intercept
acc = 2*reg.slope
obs = y
exp = linearCurve(reg.slope, reg.intercept)
chisqure_value = scipy.stats.chisquare(obs,exp).statistic


print("The accelaeration is ",acc," [m/s^2]")
print("The initail velocity is ",v0," [m/s]")
print("The R^2 value of the fit is ",reg.rvalue**2)
print("The p value of the fit is ",reg.pvalue)
print("The chi square value of the fit is ",chisqure_value)


print("\n\n\n")
print("The electron mass is ",scipy.constants.electron_mass, "[kg]")
print("Epsilon_0 value is ",scipy.constants.epsilon_0)
print("Mu_0 value is ",scipy.constants.mu_0)
