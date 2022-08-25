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
    L= 0.980 #length of pipe in [m]
    L_err = 0.001
    f_ressunance = [178,353,530,706,883,1060,1237,1413,1590,1767,2121][:-1]
    f_err= [1,1,1,1,1,1,1,1,1,1,1][:-1]
    n = list(range(1,11)) 
    
    f_ressunance = np.array(f_ressunance)
    f_err = np.array(f_err)
    n = np.array(n)

    fit1 = cfit(linearCurve_ZeroIntercept,n,f_ressunance)
    reg1 = linregress(n,f_ressunance)
    fig1=plt.figure("figure1",dpi=300)
    plt.errorbar(n,f_ressunance,f_err,fmt="o",label="Data")
    #plt.plot(n,linearCurve_ZeroIntercept(fit1[0][0], n),"-.",label="Regression")
    plt.plot(n,linearCurve(reg1.slope,reg1.intercept, n),"-.",label="Regression")
    plt.grid()
    plt.xlabel("n")
    plt.ylabel("$f [Hz]$")
    
    
    # ---- need to make sure ---
    # wave_length = 2*L/n
    # v_measured= f_ressunance*wave_length
    
    # f = (v/2L) * n
    m  = reg1.slope
    m_err = reg1.stderr
    v_measured= 2 * L * m
    v_err = 2 * np.sqrt((L*m_err)**2 + (m*L_err)**2) 
    #v_mesasured = 2 * L * reg1.slope
    T=23.4 +273.15 # temperature in kelvin
    # v_theory= speed_of_sound_calc(T,gas.gamma,gas.mass)
    v_theory = 331.7*np.sqrt((T)/273.15) #[ m/sec]
    v_relative_err= abs(v_theory-v_measured)/v_measured
    print("v theory:%fm/s"%(v_theory))
    print("v measured:%f+-%fm/s"%(v_measured,v_err))
    print("v relative error is: ", v_relative_err)
    
    # im not sure about this part
    wavelen_theory =  2 * L / n 
    f_theory = v_theory/wavelen_theory
    
    wavelen_measured = v_measured / f_ressunance
    wavelen_err = wavelen_measured * np.sqrt((v_err/v_measured)**2 + (f_err/f_ressunance)**2)
  
    
    plt.plot(n,f_theory,"d",label="Theory")
    reg2 = linregress(n,f_theory)
    plt.plot(n,f_theory,"-d",label="Theory Regression")

    plt.legend()
    plt.show()  
  
    fig2=plt.figure("figure2",dpi=300)
    plt.errorbar(n,wavelen_measured,yerr=wavelen_err,fmt="o",label="Data") # error bar will be fixed later on
    plt.plot(n,wavelen_theory,"d",label="Theory")
    plt.grid()
    plt.legend()
    plt.xlabel("n")
    plt.ylabel(r"$\lambda [m]$")
    plt.show()    


def part4_interference_and_standing_wave():
    L = 0.267 # [m]
    L_err = 0.001 
    n = np.array(list(range(1,10)))
    wavelen = 2 * L / n 
    T= 23.3 + 273.15 # temperature in kelvin
    v_theory = 331.7*np.sqrt((T)/273.15) #[ m/sec]
    f_theory = v_theory / wavelen    
    f = 3882
    f_err = 1
    
    x = [2.7,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18] #cm
    x_err = [0.1] * len(x)
  #  amplitude = [254.7,176.6,85.1,27.9,87,152.6,210,234.9,242.5] #mV
    amplitude = [297.5,204.1,79,26.5,95.12,173.8,240,263.3,259,214.7,149.5,81,18,82,156,227,267.5,255,209,152.8,78,18.5,68.5,156.5,203,229,230,203,144.5,79.3,13.4,102]
  #  amplitude_err = [0.1,0.5,0.5,0.1,0.1,0.1,0.1,0.1,0.1]
    amplitude_err = [0.1,0.1,1,0.1,0.01,0.1,1,0.1,1,0.5,0.1,1,0.5,0.5,1,1,0.2,0.2,1,0.5,1,0.2,0.5,0.5,0.5,1,1,1,0.5,0.5,0.5,1]
    
    
    amplitude = np.array(amplitude)
    amplitude_err = np.array(amplitude_err)
    x = np.array(x) * 1e-2
    x_err = np.array(x_err) *1e-2
    
    in_wavelen = 2*L / 6 # n=6 is the order
    fit_fucntion = lambda x,a,wavelen,phi0 : np.abs(2*a*np.cos(np.pi/wavelen * (2*x-L)+phi0/2))

    fit1 = cfit(fit_fucntion,x,amplitude,p0=[65,v_theory/f,0])
    fig1 = plt.figure("figure1", dpi=300)
    plt.errorbar(x,amplitude,yerr=amplitude_err,xerr=x_err,fmt="o",label="Data")
    plt.plot(x,fit_fucntion(x,*(fit1[0])),'-.',label="Regression")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("A [mV]")
    plt.grid()
    plt.show()    
    
    a,wavelen,phi0 = fit1[0][0],fit1[0][1],fit1[0][2]
    a_err,wavelen_err,phi0_err = np.sqrt(np.diag(fit1[1]))
    v = wavelen * f
    #v_err = np.sqrt()
    v_err=np.sqrt((f*wavelen_err)**2 + (f_err*wavelen)**2)
    
    print("v measured:%.1f+-%.1fm/s"%(v,v_err))
    print("relative error:",abs(v-v_theory)/v_theory,"%")
    print("wavelen: %.4f+-%.4f"%(wavelen,wavelen_err),"m")
    print("phi:%.1f+-%.1f"%(phi0*180/np.pi,phi0_err*180/np.pi),"degree")
    print("\n\n")
    
    
    x = [2.7,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5,12,12.5,13,13.5,14,14.5,15,15.5,16,16.5,17,17.5,18][4:] #cm
    x_err = [0.1] * len(x)
    amplitude = [233,255.2,82,33.3,69.3,107,160.5,181,185,173.5,137,90.4,31.5,39,94.6,141.3,182.5,188,170,131.5,81.5,31,40,89.6,143,156.5,176,173,137,90.4,30,42.9][4:]
    amplitude_err = [0.5,0.5,1,0.2,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.3,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,0.5,0.5,0.5][4:]
    x = x[:len(amplitude)]
    x_err = x_err[:len(x)]
    
    amplitude = np.array(amplitude)
    amplitude_err = np.array(amplitude_err)
    x = np.array(x) * 1e-2
    x_err = np.array(x_err) *1e-2
    
    fit_fucntion = lambda x,a,wavelen,phi0 : np.abs(2*a*np.cos(np.pi/wavelen * (2*x-L)+phi0/2))


    fit2= cfit(fit_fucntion,x,amplitude,p0=[90,v_theory/f,0])
    fig2 = plt.figure("figure2", dpi=300)
    plt.errorbar(x,amplitude,yerr=amplitude_err,xerr=x_err,fmt="o",label="Data")
    plt.plot(x,fit_fucntion(x,*fit2[0]),'-.',label="Regression")
    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("A [mV]")
    plt.grid()
    plt.show()    
    
    a,wavelen,phi0 = fit2[0][0],fit2[0][1],fit2[0][2]
    a_err,wavelen_err,phi0_err = np.sqrt(np.diag(fit2[1]))
    v = wavelen * f
    v_err=np.sqrt((f*wavelen_err)**2 + (f_err*wavelen)**2)
    print("v measured:%.1f+-%.1fm/s"%(v,v_err))
    print("relative error:",abs(v-v_theory)/v_theory,"%")
    print("wavelen: %.4f+-%.4f"%(wavelen,wavelen_err),"m")
    print("phi:%.1f+-%.1f"%(phi0*180/np.pi,phi0_err*180/np.pi),"degree")
    print("\n\n")

def main():
    print("part3:")
    part3_find_resunance_exp()
    print("\npart4:")
    part4_interference_and_standing_wave()



    



main()