# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 09:50:36 2022

@author: dorsh
"""

import numpy as np # math functions
import scipy # scientific functions
import matplotlib.pyplot as plt # for plotting figures and setting their properties
import pandas as pd # handling data structures (loaded from files)
from scipy.stats import linregress # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit # non-linear curve fitting


#%% capacitor

def V_decay(t,a,b):
    return a*np.exp(-t*b)

def linearCurve(x,a,b):
    return a*x+b

# initilazing the values
eps0 = 8.854e-12 # F/m 
D = 18e-2 # m
d = 0.5e-3 # m

C_theortical = eps0 * np.pi * D**2 / (4*d)

R_tot = 38.4e3 #ohm
R = 977 #ohm
tau_theoretical = C_theortical * R_tot


C_data = pd.read_csv('capacitor.csv')
C_data = C_data.rename(columns = {"time (sec)":"t","ch2":"V_R"})
C_data["V_C"] = C_data["ch1"] - C_data["V_R"] 

t = (C_data["t"].values) 
V_C = (C_data["V_C"].values)
plt.figure()
plt.plot(t*1e6,V_C,".",label = "V_C(t)")
plt.xlabel("Time [Us]")
plt.ylabel("V_C [V]")

# curve fitting is a methode in whitch we search a functino to fit the data best
fit2 = cfit(V_decay,t,V_C) #we are trying to determind 'a' and 'b'
plt.plot(t*1e6,V_decay(t,fit2[0][0],fit2[0][1]),label="fitted curev")
plt.legend()
plt.grid()
plt.show()
tau_exp = 1 / fit2[0][1]

plt.Figure()
plt.grid()
plt.xlabel("Time [sec]")
plt.ylabel("ln(V_C)")

plt.plot(t,np.log(V_C),'.', label = "ln(V_C(t))")
t1, t2 = 0,0.5e-4
indes = (t>t1) & (t<t2)
plt.plot(t[indes],np.log(V_C)[indes],'.', label = "ln(V_C(t)) linear")
plt.legend()
plt.show()

# A linear regresion is a curve fitting to a linear function
reg2 = linregress(t[indes],np.log(V_C)[indes])
tau_regression = -1 / reg2.slope
V0_exp = np.exp(reg2.intercept)

int_V_R = scipy.integrate.cumtrapz(C_data["V_R"],x=t,initial=0)
plt.figure()
plt.plot(int_V_R,V_C, '.',label = "V_R integration data")
plt.xlabel("integral of_V_R [V*sec]")
plt.ylabel("V_C [V]")
plt.grid()


reg3 = linregress(int_V_R,V_C)
C_meas = (1 / reg3.slope) / R
plt.plot(int_V_R,linearCurve(int_V_R, reg3.slope, reg3.intercept),'-',label="regressino")
plt.legend()
plt.show()



print("Theoretical Tau:",tau_theoretical, "[sec]")
print("Tau from exp regression:",tau_exp, "[sec]")
print("Tau from linear regression:",tau_regression,"[sec]")
print("C theoretical:",C_theortical, "[F]")
print("C measured:",C_meas,"[F]")