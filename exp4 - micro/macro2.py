# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:21:43 2022

@author: dorsh
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit




# helper function for plotting data and regression
def one4all(xdata,ydata,yerr=0,xerr=0,mode="general function",f=None,xlabel="x",ylabel="y"):
   # print(xdata,ydata,yerr,xerr,mode,f,xlabel,ylabel)
    fig = plt.figure(dpi=300)
    plt.errorbar(xdata,ydata,yerr,xerr,"o",label="Data")

    
    if mode == "none":
        fit= []
        
        
    
    elif mode == "linear":
        fit = linregress(xdata,ydata)
        f = lambda x,a,b: a*x+b
        #fit_label = "Regression: y=" + str(fit.slope) + str("x+") + str(fit.intercept)
        plt.plot(xdata,f(xdata,fit.slope,fit.intercept),"-.",label="Regression")
    
       
        
    elif mode == "0 intercept":
        f = lambda x,a: a*x
        fit = cfit(f,xdata,ydata)
        plt.plot(xdata,f(xdata,*fit[0]),"-.",label="Regression")
        
    elif mode == "general function":
        if f==None:
            raise TypeError

        fit = cfit(f,xdata,ydata)
        plt.plot(xdata,f(xdata,*fit[0]),"-.",label="Regression")
    
    
    else:
        print("mode='",mode,"' is not definted!")
        raise TypeError


    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid()
    if mode != "none":
        plt.legend()

    plt.show()
    
    return (fig,fit)



#%% 3 michelson


x_node = [1,1] #m
x_max = [1,1]
x_err = 0

V_node = [1,1]
V_max = [1,1]
V_err = 0


x_node = np.array(x_node)
x_max = np.array(x_max)
V_node = np.array(V_node)
V_max = np.array(V_max)

wavelen_node = 2 * (np.abs(np.diff(x_node))).mean()
wavelen_node_err = 2 * np.sqrt(2) * x_err / np.sqrt(len(x_node) - 1)
print('wavelen node:',wavelen_node,"+-",wavelen_node_err,"m")

wavelen_max = 2 * (np.abs(np.diff(x_max))).mean()
wavelen_max_err = 2* np.sqrt(2) * x_err / np.sqrt(len(x_max) - 1)
print('wavelen max:',wavelen_max,"+-",wavelen_max_err,"m")


wavelen = np.array([wavelen_node,wavelen_max]).mean()
wavelen_err = np.sqrt(wavelen_node_err**2+wavelen_max_err**2)
print('wavelen:',wavelen,"+-",wavelen_err,"m")

x = np.append(x_node,x_max)
V = np.append(V_node,V_max)
fig = plt.figure()
one4all(x,V,V_err,x_err,"none",xlabel="x [m]",ylabel="V [V]")

#%% 4 ferbri febro 

d1 = 0
d2 = 0

x_node = [1,1] #m
x_err = 0

V_node = [1,1]
Vf_err = 0


x_node = np.array(x_node)
V_node = np.array(V_node)

wavelen_node = 2 * (np.abs(np.diff(x_node))).mean()
wavelen_node_err = 2 * np.sqrt(2) * x_err / np.sqrt(len(x_node) - 1)
print('wavelen 1:',wavelen_node,"+-",wavelen_node_err,"m")


#### part 2 
d1 = 0
d2 = 0

x_node = [1,1] #m
x_err = 0

V_node = [1,1]
V_err = 0


x_node = np.array(x_node)
V_node = np.array(V_node)

wavelen_node2 = 2 * (np.abs(np.diff(x_node))).mean()
wavelen_node_err2 = 2 * np.sqrt(2) * x_err / np.sqrt(len(x_node) - 1)
print('wavelen 2:',wavelen_node2,"+-",wavelen_node_err2,"m")


wavelen = np.array([wavelen_node,wavelen_node2]).mean()
wavelen_err = np.sqrt(wavelen_node_err**2+wavelen_node_err2**2)
print('wavelen:',wavelen,"+-",wavelen_err,"m")

#%% 5 Loyd
d1= 1 # distance of first max observed
h = np.array([1])
h_err = 0.001 #m
v = np.array([1]) 
v_err = np.array([1])
h1 = 1
h2 = 2
# notice V and I !!!!!
fig,fit = one4all(h,v,v_err,h_err,"none",xlabel="x [m]",ylabel="V [V]")
wavelen = 2 * (np.sqrt(h2**2 + d1**2) - np.sqrt(h1**2 + d1**2))
wavelen_err = 2 * np.sqrt((h2*h_err/np.sqrt(h2**2+d1**2))**2+(h1*h_err/np.sqrt(h2**2+d1**2))**2+(d1*h_err/np.sqrt(h2**2+d1**2))**2)
print('wavelen:',wavelen,"+-",wavelen_err,"m")


phi = (2*np.sqrt(h**2+d1**2)-2*d1) * 2 * np.pi / wavelen + np.pi
cos_phi = np.cos(phi)
(fig,fit)=one4all(cos_phi,v,mode="linear",xlabel=r"$cos(\theta)$",ylabel="$V [V]$")



### part 2
d1= 1 # distance of first max observed
h = np.array([1])
h_err = 0.001 #m
v = np.array([1]) 
v_err = np.array([1])
h1 = 1
h2 = 2
# notice V and I !!!!!
fig,fit = one4all(h,v,v_err,h_err,"none",xlabel="x [m]",ylabel="V [V]")
wavelen = 2 * (np.sqrt(h2**2 + d1**2) - np.sqrt(h1**2 + d1**2))
wavelen_err = 2 * np.sqrt((h2*h_err/np.sqrt(h2**2+d1**2))**2+(h1*h_err/np.sqrt(h2**2+d1**2))**2+(d1*h_err/np.sqrt(h2**2+d1**2))**2)
print('wavelen:',wavelen,"+-",wavelen_err,"m")


phi = (2*np.sqrt(h**2+d1**2)-2*d1) * 2 * np.pi / wavelen + np.pi
cos_phi = np.cos(phi)
(fig,fit)=one4all(cos_phi,v,mode="linear",xlabel=r"$cos(\theta)$",ylabel="$V [V]$")





