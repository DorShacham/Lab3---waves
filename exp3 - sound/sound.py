# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 08:47:10 2022

@author: Lab3
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit
from uncertainties import unumpy
from uncertainties import ufloat
from uncertainties.umath import *



def linearCurve(a,b,x):
    return a*x+b
def linearCurve_ZeroIntercept(a,x):
    return linearCurve(a,0,x)

def speed_of_sound_calc(T,gamma,M,R=8.31):
    return sqrt(gamma*R*(T+273)/M)

# part 0 
def part0_get_to_know_the_equipment_exp():
    f = [ 3303 , 3600,  4000 , 5000 , 4502, 4103, 5498 , 6000, 6498, 7001] # hz
    f_err = [ 1 , 10 ,1, 1, 1, 1, 1, 1, 1, 1] # may be obislite
    amplitude = [3.7 , 8.2, 46, 24, 25.5, 87, 65, 45, 57, 41] #mV
    amplitude_err = [0.1 , 0.1, 2, 0.1 , 0.5, 1 ,1 ,1, 1, 1]
    
    plt.figure("figure0",dpi=300)
    plt.errorbar(f,amplitude,yerr=amplitude_err,xerr=f_err,fmt='o')
    plt.grid()
    plt.xlabel("f [Hz]")
    plt.ylabel("Amplitude [mV]")
    plt.show()
    
    '''
    explain results:
    '''
    
    

#air speed of sound experiment
def part1_speed_of_sound_air_exp():
    x0=43
    #x0 of buzzer is 30cm
    '''
    previous results
    dphi=[ 0 , 180 , 360 , 540 , 720] # degrees
    dphi_err=[ 2 , 1,  ]
    dx=[35 , 40 ] # [cm]
    dx_err=[0.5, 0.5  ] #[cm]
    '''
    
    f=ufloat(4103,1) # [Hz] --- update f
    dphi=[ 0 , 180 , 360 , 540 , 720 , 900] # degrees
    dphi_err=[ 2, 5 , 5 ,  5 , 5 ,5]
    dx=[43 , 47 , 50.9, 55, 58.5 ,63.5] # [cm]
    dx_err=[0.5, 0.5,  0.5 , 0.5 ,0.5 ,0.5] #[cm]
    dphi = np.array(dphi) * np.pi /180
    dphi_err = np.array(dphi_err) * np.pi/180
    dphi = unumpy.uarray(dphi,dphi_err)
    
    dx=(np.array(dx)-x0) *1e-2
    dx_err=np.array(dx_err) *1e-2
    dx = unumpy.uarray(dx,dx_err)

    #dx_err=np.array([]) - why?
    #fit2 = cfit(linearCurve_ZeroIntercept,dx,dphi)
    fit1 = linregress(unumpy.nominal_values(dx),unumpy.nominal_values(dphi))
    slope = ufloat(fit1.slope,2*fit1.stderr)
    lambda_regression=2*np.pi/slope
    fig1=plt.figure("figure1",dpi=300)
    plt.errorbar(unumpy.nominal_values(dx),unumpy.nominal_values(dphi),yerr=unumpy.std_devs(dphi),xerr=unumpy.std_devs(dx),fmt='o',label="Data")
    plt.plot(unumpy.nominal_values(dx),linearCurve(fit1.slope,fit1.intercept,unumpy.nominal_values(dx)),"-.",label="Regression")
    #plt.plot(dx[:-1],linearCurve_ZeroIntercept(fit1[0][0],dx[:-1]),"-.",label="Regression")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\Delta x[m]$")
    plt.ylabel(r"$\Delta\phi[rad]$")
    plt.show()

    print("the wave length is: ", lambda_regression , " [m]")
    v_air=lambda_regression*f
    print("the speed of sound measured is ", v_air , " [m/sec]")
    T = ufloat(23.1,0.1) #temperature in the lab in Celcius
    v_air_theory = ufloat(331.7,0.1)*sqrt((T+273.15)/273.15) #[ m/sec]
    v_air_relative_error= abs(v_air_theory-v_air)/v_air_theory
    print("The relative error is: ", (v_air_relative_error.n+v_air_relative_error.s)*100,"%")



def part2_speed_of_sound_gas_exp():
    L=0.980 # length of pipe in [m] 
    L_err=0.001
    L = ufloat(L,L_err)
    
    T = ufloat(22.4,0.1) # update
    print("f air < 165.810 Hz")
    print("f He < 514.960 Hz")
    print("f CO2 < 137.613 Hz")
    print("f is a upper bound")

    # we can calculte the mean value of the first and the secondry wave hit
    # air
    t_air = 2.946e-3 # sec
    t_air_err = 0.01e-3
    t_air = ufloat(t_air,t_air_err)
  
    v_air = L / t_air
    print("\nthe speed of sound measured in air is ", v_air , " [m/sec]")
    v_air_theory = ufloat(331.7,0.1)*sqrt((T+273.15)/273.15) #[ m/sec]
    v_air_relative_error= abs(v_air_theory-v_air)/v_air_theory
    print("The relative error is: ", (v_air_relative_error.n+v_air_relative_error.s)*100,"%")
    # He
    t_He = 1.009e-3
    t_He_err = 0.01e-3
    t_He = ufloat(t_He,t_He_err)
    
    v_He = L / t_He
    print("\nthe speed of sound measured in He is ", v_He , " [m/sec]")
    v_He_theory = speed_of_sound_calc(T,1.66,4e-3) #[ m/sec]
    v_He_relative_error= abs(v_He_theory-v_He)/v_He_theory
    print("The relative error is: ", (v_He_relative_error.n+v_He_relative_error.s)*100,"%")
    
    # CO2
    t_CO2 = 3.74e-3
    t_CO2_err = 0.1e-3
    t_CO2 = ufloat(t_CO2,t_CO2_err)
  
    v_CO2 = L / t_CO2
    print("\nthe speed of sound measured in CO2 is ", v_CO2 , " [m/sec]")
    v_CO2_theory = speed_of_sound_calc(T,1.304,44e-3) #[ m/sec]
    v_CO2_relative_error= abs(v_CO2_theory-v_CO2)/v_CO2_theory
    print("The relative error is: ", (v_CO2_relative_error.n+v_CO2_relative_error.s)*100,"%")
    

    return

class Gas:
    def __init__(mass,gamma):
        self.mass=mass
        self.gamma=gamma
    
        
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
   part1_speed_of_sound_air_exp()



    



main()