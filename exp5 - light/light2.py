import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit
from scipy.optimize import curve_fit as cfit


def one4all(xdata,ydata,yerr=0,xerr=0,mode="general function",f=None,xlabel="x",ylabel="y",show=True):
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
        
    if show:
        plt.show()
    
    return (fig,fit)

def Rsqrue(x,y):
    RSS = np.sum((y-x)**2)
    TSS = np.sum(y**2)
    return 1 - RSS/TSS

def Reg_print(fit):
    m = ufloat(fit.slope,fit.stderr*2)
    b = ufloat(fit.intercept,fit.intercept_stderr*2)
    print("==> y =(",m,")x + (",b,") , R^2=",fit.rvalue**2)
    
    
def print_seciont(str):
    print("\n\n==================",str,"====================\n\n")
    
#%% init
T_env = 0 + 273.15 # in Kelvin!!
T_env_err = 1
p_env  = 1
P_env_err = 1
L = 1
L_err = 1
wavelen0 = 1
kb = scipy.constants.k # Boltzmann const in J/k

p_nom = 76 # cmHg
T_nom = 273.15 # K, 0 C


#%% air
print_seciont("Air")
p = np.array([0,1]) # need to notice units and do conversion to cmHg or other the other way around
p_err = np.array([0.1,0.1])
F = np.arange(0,len(p)) * 10 # lines that passes over the screen

fig,fit = one4all(p, F,0,p_err,"linear",None,"p [pressure unit]","F")
m = fit.slope
alpha = m * 2 * kb * T_env * wavelen0 / L
print("Alpha air is:",alpha)

n = 1 + alpha * p_nom / (2 * kb * T_nom * wavelen0)
print("n air is:",n)

#%% CO2
print_seciont("CO2")
p = np.array([0,1]) # need to notice units and do conversion to cmHg or other the other way around
p_err = np.array([0.1,0.1])
F = np.arange(0,len(p)) * 10 # lines that passes over the screen

fig,fit = one4all(p, F,0,p_err,"linear",None,"p [pressure unit]","F")
m = fit.slope
alpha = m * 2 * kb * T_env * wavelen0 / L
print("Alpha CO2 is:",alpha)

n_CO2 = 1 + alpha * p_nom / (2 * kb * T_nom * wavelen0)
print("n CO2 is:",n_CO2)


#%% He
print_seciont("He")
p = np.array([0,1]) # need to notice units and do conversion to cmHg or other the other way around
p_err = np.array([0.1,0.1])
F = np.arange(0,len(p)) * 10 # lines that passes over the screen

fig,fit = one4all(p, F,0,p_err,"linear",None,"p [pressure unit]","F")
m = fit.slope
alpha = m * 2 * kb * T_env * wavelen0 / L
print("Alpha He is:",alpha)

n_He = 1 + alpha * p_nom / (2 * kb * T_nom * wavelen0)
print("n He is:",n_He)


#%% Mixture of CO2 and He
print_seciont("Mixture of CO2 and He")
p = np.array([0,1]) # need to notice units and do conversion to cmHg or other the other way around
p_err = np.array([0.1,0.1])
F = np.arange(0,len(p)) * 10 # lines that passes over the screen

fig,fit = one4all(p, F,0,p_err,"linear",None,"p [pressure unit]","F")
m = fit.slope
alpha = m * 2 * kb * T_env * wavelen0 / L
print("Alpha mixture is:",alpha)

n = 1 + alpha * p_nom / (2 * kb * T_nom * wavelen0)
print("n mixture is:",n)

CO2overHe = (n_He - n)/(n - n_CO2)
CO2_part = CO2overHe * 100 #percent
He_part = 0 # to be contintinued
print("There is %.2f%% CO2 and %.2f%% He in the mixture"%(CO2_part,He_part))