# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 17:02:24 2022

@author: dorsh
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit

#%% wide unit
def liner_curve(x,a,b):
    return a*x+b

def liner_fit(x,a):
    return a*x

alpha = 0.11 #rad/bar
F = 0.49 # N
l = 45.6e-2 #meter
m = 43.2e-3 # kg/bar
N = 72
d = 1.27e-2 # meter/bar

k = F*l / (2*alpha)


I = m*(l**2)/12 
v = d*(k/I)**0.5  # m/s (with out d is bar/sec)
L =N * d

# holded edge
print("holded edge:")
for n in    range(1,9):
    f = n * v / (2*L)
    print("f(%d)=%f"%(n,f))
    
exp_f_holded = np.array([0.303,0.529,0.778,1.01,1.265,1.449,1.694])
n = np.array(range(1,8))
plt.figure()
plt.errorbar(n,exp_f_holded,yerr=1e-4,fmt='o',label="Data")
plt.plot(n,liner_curve(n, v / (2*L), 0),label="Theoretical curve")
plt.grid()

plt.xlabel("n - Index of standing wave")
plt.ylabel("f [Hz] - frequency of the engine ")


#reg_holded = linregress(n,exp_f_holded)
#plt.plot(n,liner_curve(n,reg_holded.slope, reg_holded.intercept),linestyle='-.',label='Linear regression',color='black')
fit = cfit(liner_fit,n,exp_f_holded)
plt.plot(n,liner_fit(n,fit[0]),linestyle='-.',label='Linear regression',color='black')
print("\nHolded edge: a=%f, b=0"%(fit[0]))

err_holded =(np.sum((exp_f_holded-liner_curve(n, v / (2*L), 0))**2))**0.5
print("LS error:",err_holded)
a_holded_err = abs(fit[0] - v / (2*L)) / (v / (2*L)) *100
print("regression error:%f%%"%(a_holded_err))

plt.legend()
plt.show()


    
# free edge
print("\nfree edge:")
for n in range(1,10):
    f = (2*n-1)*v/(4*L)
    print("f(%d)=%f"%(n,f))
    
exp_f_free  = np.array([0.401,0.645,0.917,1.09,1.449,1.694,2])
n = np.array(range(2,9))
plt.figure()
plt.errorbar(n,exp_f_free,yerr=1e-4,fmt='o',label="Data")
plt.plot(n,liner_curve(n, 2*v/(4*L), -v/(4*L)),label="Theoretical curve")
plt.grid()

plt.xlabel("n - Index of standing wave")
plt.ylabel("f [Hz] - frequency of the engine ")

reg_free = linregress(n,exp_f_free)
plt.plot(n,liner_curve(n,reg_free.slope, reg_free.intercept),'-.',label='Linear regression',color='black')
print("\nfree edge: a=%f, b=%f\nR-squre value:%f"%(reg_free.slope,reg_free.intercept,reg_free.rvalue**2))

err_free =(np.sum((exp_f_free-liner_curve(n, 2*v/(4*L), -v/(4*L)))))**0.5
print("LS error:",err_free)
a_free_err = abs(reg_free.slope - v / (2*L)) / (v / (2*L)) *100
print("a regression error:%f%%"%(a_free_err))
b_free_err = abs(reg_free.intercept - -v/(4*L)) / (v/(4*L)) *100
print("b regression error:%f%%"%(b_free_err))

plt.legend()
plt.show()


