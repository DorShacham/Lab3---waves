# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 17:16:33 2022

@author: dorsh
"""

import numpy as np # math functions
import scipy # scientific functions
import matplotlib.pyplot as plt # for plotting figures and setting their properties
import pandas as pd # handling data structures (loaded from files)
from scipy.stats import linregress # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit # non-linear curve fitting

#%% potential

def potential(x, y, a, C): #calculte the potential
    arg1 = np.sqrt(np.power(x-a,2)+np.power(y,2))/a #calculate arg for the first log
    arg2 = np.sqrt(np.power(x+a,2)+np.power(y,2))/a #calculate arg for the second log
    pot = C * (-np.log(arg1)+np.log(arg2)) #calculte everything toghther
    return pot #return the potential


C = 1 #initilazing the values
a = 1
L = 3
N = 100
coord = np.linspace(-L, L , N) # defines coordinates 
coord_x, coord_y = np.meshgrid(coord, coord) #creating a meshgrid for x and y cord
V_xy = potential(coord_x, coord_y, a, C) # calculte the value of the potential 

plt.figure() #create a figure object
plt.pcolormesh(coord_x, coord_y, V_xy) #create a colorful plot
plt.colorbar() # adds a color bar
plt.contour(coord_x, coord_y, V_xy, np.sort(np.linspace(-5,5)), cmap='hot') #The cmap defines the range of colors 
plt.show() #revile the plot 


x = coord
y = np.zeros(N)
V_x = potential(x, y, a, C)
plt.plot(x,V_x,'.',label="calculated potential")

#%% capacitor
# initilazing the values
eps0 = 8.854e-12 # F/m 
D = 18e-2 # m
d = 0.5e-3 # m

C_theortical = eps0 * np.pi * D**2 / (4*d)
print(C_theortical) #in F

R_tot = 38.4e3 #ohm
R = 977 #ohm
tau_theoretical = C_theortical * R_tot
print(tau_theoretical) # in sec


C_data = pd.read_csv('capacitor.csv')
C_data = C_data.rename(columns = {"time (sec)":"t","ch2":"V_R"})
C_data["V_C"] = C_data["ch1"] - C_data["V_R"] 

t = (C_data["t"].values) * 1e-6
V_C = np.array(C_data["V_C"].values)
plt.figure()
plt.plot(t,V_C,".",label = "V_C(t)")
plt.xlabel("Time [us]")
plt.ylabel("V_C [V]")
plt.show()
