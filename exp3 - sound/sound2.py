# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:59:35 2022

@author: dorsh
"""


import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit




def linearCurve(a,b,x):
    return a*x+b
def linearCurve_ZeroIntercept(a,x):
    return linearCurve(a,0,x)

def speed_of_sound_calc(T,gamma,M,R=8.31):
    return sqrt(gamma*R*(T+273)/M)


def part3_find_resunance_exp():
    L= 1 #length of pipe in [m]
    f_ressunance = []
    f_err= []
    n = list(range(1,11)) # what does the order n means?
    
    f_ressunance = np.array(f_ressunance)
    f_err = np.array(f_err)
    n = np.array(n)

    fit1 = cfit(linearCurve_ZeroIntercept,n,f_ressunance)
    reg1 = linregress(n,f_ressunance)
    fig1=plt.figure("figure1")
    plt.errorbar(n,f_ressunance,f_err,fmt="o",label="Data")
    plt.plot(n,linearCurve_ZeroIntercept(fit1[0][0], n),"-.",label="Regression")
    plt.grid()
    plt.xlabel("n")
    plt.ylabel("$f [Hz]$")
    
    
    # ---- need to make sure ---
    # wave_length = 2*L/n
    # v_measured= f_ressunance*wave_length
    
    # f = (v/2L) * n
    v_measured= 2 * L * fit1[0][0]
    #v_mesasured = 2 * L * reg1.slope
    T=293 # temperature in kelvin
    # v_theory= speed_of_sound_calc(T,gas.gamma,gas.mass)
    v_theory = 331.7*np.sqrt((T)/273.15) #[ m/sec]
    v_relative_err= (v_theory-v_measured)/v_measured
    print("v relative error is: ", v_relative_err)
    
    # im not sure about this part
    wavelen_theory =  2 * L / n 
    f_theory = v_theory/wavelen_theory
    
    wavelen_measured = v_measured / f_ressunance
  
    
    plt.plot(n,f_theory,"-",label="Theory")
    plt.legend()
    plt.show()  
  
    fig2=plt.figure("figure2",dpi=300)
    plt.errorbar(n,wavelen_measured,0,fmt="o",label="Data") # error bar will be fixed later on
    plt.plot(n,wavelen_theory,"-",label="Theory")
    plt.grid()
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("$\lambda [m]$")
    plt.show()    


def part4_interference_and_standing_wave():
    L = 0 # [m]
    L_err = 0 
    n = np.array(list(range(10)))
    wavelen = 2 * L / n 
    T=293 # temperature in kelvin
    v_theory = 331.7*np.sqrt((T)/273.15) #[ m/sec]
    
    f = 0
    f_err = 0
    
    amplitude = []
    amplitude_err = []
    x = []
    x_err = []
    
    np.array(amplitude)
    np.array(amplitude_err)
    np.array(x)
    np.array(x_err)
    
    fit_fucntion = lambda x,a,wavelen,L,phi0 : 2*a*np.cos(np.pi/wavelen * (2*x-L)+phi0/2)
    fit1 = cfit(fit_fucntion,x,amplitude)
    fig1 = plt.figure("figure1", dpi=300)
    plt.errorbar(x,amplitude,yerr=amplitude_err,xerr=x_err,fmt="o",label="Data")
    plt.plot(x,fit_function(x,*fit1[0]),'-.',label="Regression")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("A [mV]")
    plt.show()    
    
    v = fit1[0][2] * np.mean(f1,f2)
    print("v measured:",v,"m/s")
    
    
    
    
    amplitude = []
    amplitude_err = []
    x = []
    x_err = []
    
    np.array(amplitude)
    np.array(amplitude_err)
    np.array(x)
    np.array(x_err)
    
    fit_fucntion = lambda x,a,wavelen,L,phi0 : 2*a*np.cos(np.pi/wavelen * (2*x-L)+phi0/2)
    fit2= cfit(fit_fucntion,x,amplitude)
    fig2 = plt.figure("figure2", dpi=300)
    plt.errorbar(x,amplitude,yerr=amplitude_err,xerr=x_err,fmt="o",label="Data")
    plt.plot(x,fit_function(x,*fit1[0]),'-.',label="Regression")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("A [mV]")
    plt.show()    
    
    v = fit1[0][2] * np.mean(f1,f2)
    print("v measured:",v,"m/s")

def main():
   part3_find_resunance_exp()



    



main()