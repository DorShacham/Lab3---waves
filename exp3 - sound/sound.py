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
    return np.sqrt(gamma*R*T/M)

# part 0 
def part0_get_to_know_the_equipment_exp():
    f = []
    f_err = [] # may be obislite
    amplitude = []
    amplitude_err = []
    
    plt.figure("figure0",dpi=300)
    plt.errorbar(f,amplitude,yerr=amplitude_err,xerr=f_err,fmt='o')
    plt.grid()
    plt.xlabel("f [Hz]")
    plt.ylabel("Amplitude [m]")
    plt.show()
    
     '''
    explain results:


    '''
    
    

#air speed of sound experiment
def part1_speed_of_sound_air_exp():
    dphi=[]
    dphi_err=[]
    dx=[] # [m]
    dx_err=[] #[m]
    f=1 # [Hz] --- update f
    
    dphi = np.array(dphi)
    dphi_err = np.array(dphi_err)
    dx=np.array(dx)
    dx_err=np.array(dx_err)

    #dx_err=np.array([]) - why?
    fit1 = cfit(linearCurve_ZeroIntercept,dx,dphi)
    lambda_regression=2*np.pi/fit1[0]
    fig1=plt.figure("figure1",dpi=300)
    plt.errorbar(dx,dphi,yerr=dphi_err,xerr=dx_err,fmt='o',label="Data")
    plt.plot(dx,linearCurve_ZeroIntercept(fit1[0],dx),fmt="-.",label="Regression")
    plt.grid()
    plt.legend()
    plt.xlabel(r"$\Delta x[m]$")
    plt.ylabel(r"$\Delta\phi[rad]$")
    plt.show()

    print("the wave length is: ", lambda_regression , " [m]")
    v_air=lambda_regression*f
    print("the speed of sound measured is ", v , " [m/sec]")
    T = 20 #temperature in the lab in Celcius
    v_air_theory = 313.7*np.sqrt((T+273)/273) #[ m/sec]
    v_air_relative_error= abs(v_air_theory-v_air)/v_air_theory
    print("The relative error is: ", v_air_relative_error*100)



def part2_speed_of_sound_gas_exp():
    L=1 # length of pipe in [m] 
    T = 20 # update
    print("f air > 687.26 Hz")
    print("f He > 2010.94 Hz")
    print("f CO2 > 537.386 Hz")
    print("f is a lower bound")

    # we can calculte the mean value of the first and the secondry wave hit
    # air
    t_air = 0 # sec
    t_air_err = 0
  
    v_air = L / t_air
    print("\nthe speed of sound measured in air is ", v_air , " [m/sec]")
    v_air_theory = 313.7*np.sqrt((T+273)/273) #[ m/sec]
    v_air_relative_error= abs(v_air_theory-v_air)/v_air_theory
    print("The relative error is: ", v_air_relative_error*100)
    
    # He
    t_He = 0 # sec
    t_He_err = 0
  
    v_He = L / t_He
    print("\nthe speed of sound measured in He is ", v_He , " [m/sec]")
    v_He_theory = speed_of_sound_calc(T,1.66,4e-3) #[ m/sec]
    v_He_relative_error= abs(v_He_theory-v_He)/v_He_theory
    print("The relative error is: ", v_He_relative_error*100)
    
    # CO2
    t_CO2 = 0 # sec
    t_CO2_err = 0
  
    v_CO2 = L / t_CO2
    print("\nthe speed of sound measured in CO2 is ", v_CO2 , " [m/sec]")
    v_CO2_theory = speed_of_sound_calc(T,1.304,44e-3) #[ m/sec]
    v_CO2_relative_error= abs(v_CO2_theory-v_CO2)/v_CO2_theory
    print("The relative error is: ", v_CO2_relative_error*100)
    

    return

class Gas:
    def __init__(mass,gamma):
        self.mass=mass
        self.gamma=gamma
    
        
def part3_find_resunance_exp(gas):
    L= 1 #length of pipe in [m]
    f_ressunance = []
    f_err=[]
    n = list(range(10)) # what does the order n means?
    
    f_ressunance = np.array(f_ressunance)
    f_err = np.array(f_err)
    n = np.array(n)

    fit1 = cfit(linearCurve_ZeroIntercept(a, x),n,f_ressunance)
    fig1=plt.figure("figure1")
    plt.errorbar(n,f_ressunance,f_err,fmt="o",label="Data")
    plt.grid()
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("$f_{ressunance}[Hz]$")
    plt.show()
    
    # ---- need to make sure ---
    # wave_length = 2*L/n
    # v_measured= f_ressunance*wave_length
    
    # f = (v/2L) * n
    v_mesasured = 2 * L * fit1[0][0]
    T=293 # temperature in kelvin
    # v_theory= speed_of_sound_calc(T,gas.gamma,gas.mass)
    v_theory = 313.7*np.sqrt((T)/273) #[ m/sec]
    v_relative_err= (v_theory-v_measured)/v_measured
    print("v relative error is: ", v_relative_err)
    
    # im not sure about this part
    wavelen_theory =  2 * L / n 
    wavelen_measured = v / f_ressunance
  
    fig2=plt.figure("figure2",dpi=300)
    plt.errorbar(n,wavelen_measured,0,fmt="o",label="Data") # error bar will be fixed later on
    plt.plot(n,wavelen_theory,fmt="-.",label="Theory")
    plt.grid()
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("$\lambda[m]$")
    plt.show()    
def 

def part4_interference_and_standing_wave():
    L = 0 # [m]
    L_err = 0 
    n = np.array(list(range(10)))
    wavelen = 2 * L / n 
    
    f1,f2 = 0,0
    f1_err,f2_err = 0,0
    
    amplitude = []
    amplitude_err = []
    x = []
    x_err = []
    
    np.array(amplitude)
    np.array(amplitude_err)
    np.array(x)
    np.array(x_err)
    
    fig1 = plt.figure("figure1", dpi=300)
    plt.errorbar(x,amplitude,yerr=amplitude_err,xerr=x_err,fmt="o",label="Data")
    # to be finished

def main():
    speed_of_sound_air()

    
    #



    


    return


main