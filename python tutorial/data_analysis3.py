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


#%% Ohm
def I_R(V2,R1):
    return V2/R1

def V_R(V1,V2):
    return V1 - V2

def R_t(V_R,I_R):
    return V_R/I_R

def P_t(V_R,I_R):
    return V_R*I_R

def Energy(P_t,t):
    return scipy.integrate.cumtrapz(P_t,x=t,initial=0)

def linearCurve(x,a,b):
    return a*x+b

R1 = 5.48 #ohm
R_data = pd.read_csv("ohm.csv",header=1, usecols=["Time (s)", "1 (VOLT)", "2 (VOLT)"]) #header = 1 means that the line 1 (the second line) is the header
t = R_data["Time (s)"]
V1, V2 = R_data["1 (VOLT)"], R_data["2 (VOLT)"]

I_R = I_R(V2,R1)
V_R = V_R(V1,V2)
R_t = R_t(V_R,I_R)
P_t = P_t(V_R,I_R)
E = Energy(P_t, t)

indes = (E>0.1) & (E<0.9)

plt.figure()
plt.plot(E[indes],R_t[indes],'.',label="data")
plt.grid()
plt.xlabel("Energy[J]")
plt.ylabel("R_t[Ohm]")


reg = linregress(E[indes],R_t[indes])
plt.plot(E[indes],linearCurve(E[indes], reg.slope, reg.intercept),'-',label="regression")
plt.legend()
plt.show()

R0 = reg.intercept
alpha_over_C_heat = reg.slope / R0
print("R0=",R0,"[Ohm]")
print("alpha/C_heat=",alpha_over_C_heat)
