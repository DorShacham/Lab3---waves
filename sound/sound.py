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

#air speed of sound experiment
def speed_of_sound_air_exp():
    dphi=[]
    dphi_err=[]
    dx=[] # [m]
    dx_err=[] #[m]
    f=1 # [Hz]
    
    dphi = np.array(dphi)
    dphi_err = np.array(dphi_err)
    dx=np.array(dx)
    dx_err=np.array(dx_err)

    dx_err=np.array([])
    fit1 = cfit(linearCurve_ZeroIntercept,dx,dphi)
    lambda_regression=2*np.pi/fit1[0]
    fig1=plt.figure("figure1",dpi=300)
    plt.errorbar(dx,dphi,dphi_err,dx_err,fmt='o',label="data")
    plt.plot(dx,linearCurve_ZeroIntercept(fit1[0],dx),fmt="-.",label="regression")
    plt.grid()
    plt.legend()
    plt.xlabel("dx[mm]")
    plt.ylabel("dphi")
    plt.show()

    print("the wave length is: ", lambda_regression , " [m]")
    v_air=lambda_regression*f
    print("the speed of sound measured is ", v , " [m/sec]")
    T = 20 #temperature in the lab in Celcius
    v_air_theory = 313.7*np.sqrt((T+273)/273) #[ m/sec]
    v_air_relative_error= (v_air_theory-v_air)/v_air_theory
    print("The relative error is: ", v_air_relative_error*100)

def speed_of_sound_calc(T,gamma,M,R=8.31):
    return np.sqrt(gamma*R*T/M)

def speed_of_sound_gas_exp():
    L=1 # length of pipe in [m] 

    return

class Gas:
    def __init__(mass,gamma):
        self.mass=mass
        self.gamma=gamma
    
        
def find_resunance_exp(gas):
    L= 1 #length of pipe in [m]
    f_ressunance = []
    f_err=[]
    n = list(range(10))
    n = np.array(n)

    fig1=plt.figure("figure1")
    plt.errorbar(n,f_ressunance,f_err,fmt="o",label="data")
    plt.grid()
    plt.legend()
    plt.xlabel("n")
    plt.ylabel("f_ressunance")
    plt.show()

    wave_length = 2*L/n
    v_measured= f_ressunance*wave_length
    T=293 # temperature in kelvin
    v_theory= speed_of_sound_calc(T,gas.gamma,gas.mass)
    v_relative_err= (v_theory-v_measured)/v_measured
    print("v relative error is: ", v_relative_err)



def get_to_know_the_equipment_exp():
     f = [] #hz
     f_err =[]
     A = [] #amplitude
     A_err = []

    fig1=plt.figure("figure 1",dpi=300)
    plt.errorbar(f,A,A_err,f_err,fmt="o",label="data")
    '''
    explain results:


    '''
    
def 



def main():
    speed_of_sound_air()

    
    #



    


    return


main