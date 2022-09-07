import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit as cfit
from scipy.optimize import curve_fit as cfit

from uncertainties import unumpy
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties.unumpy import nominal_values as uval
from uncertainties.unumpy import std_devs as uerr


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
T_env = 24.2 + 273.15 # in Kelvin!!
T_env_err = 1
T_env = ufloat(T_env,T_env_err)

p_env  = 761 #torr
p_env_err = 1
p_env = ufloat(p_env,p_env_err)

L = 24.8e-2 #m
L_err = 1e-3 #m
L = ufloat(L,L_err)

wavelen0 = 532e-9 #m
kb = scipy.constants.k # Boltzmann const in J/k

p_nom = 760 # mmHg
T_nom = 273.15 # K, 0 C


#%% air
print_seciont("Air")
p = np.array([ 426, 489,537,605,670,737]) # need to notice units and do conversion to cmHg or other the other way around
p_err = 1#np.array([0.1,0.1])
F = np.arange(0,len(p)) * 10 # lines that passes over the screen

fig,fit = one4all(p, F,0,p_err,"linear",None,"p [mmHg]","F")
m = ufloat(fit.slope,2*fit.stderr)
alpha = m * 2 * kb * T_env * wavelen0 / L
print("Alpha air is:",alpha)

n = 1 + alpha * p_nom / (2 * kb * T_nom )
print("n air is:",n)
Reg_print(fit)
fig.savefig("fig/plot_air")

#%% CO2
p_env= ufloat(761,1)
print_seciont("CO2")
p = np.array([30,57,92,99,103,108,129,133,151,275,343,433,474,516,553,589,632,688,740,760])[-9:] # need to notice units and do conversion to cmHg or other the other way around
p_err = 1
F= np.array([0,2,8,9,10,11,16,17,24,39,52,65,76,87,98,108,119,129,142,147])[-9:]
#F = np.arange(0,len(p)) * 10 # lines that passes over the screen

fig,fit = one4all(p, F,0,p_err,"linear",None,"p [mmHg]","F")
m = ufloat(fit.slope,2*fit.stderr)
alpha = m * 2 * kb * T_env * wavelen0 / L
print("Alpha CO2 is:",alpha)

n_CO2 = 1 + alpha * p_nom / (2 * kb * T_nom )
print("n CO2 is:",n_CO2)
Reg_print(fit)
fig.savefig("fig/plot_CO2")
#%% He
print_seciont("He")

#p = np.array([34,41,49,55]) # need to notice units and do conversion to cmHg or other the other way around

#first measurements
#*****p = np.array([400,502,558,586,615,651,752])[:-1] # need to notice units and do conversion to cmHg or other the other way around
#F_del=np.array([3,2,1,1,2])
#****F=np.array([0,3,5,6,7,9])

p_env= ufloat(762,1)
p = np.array([32,41,60,78,137,175,193,204,304,380,414,557,616,662,697]) # need to notice units and do conversion to cmHg or other the other way around
F_del=np.array([])
F=np.array([0,1,3,5,8,10,11,12,22,25,27,32,34,35,36])
p_err = 1
#F = np.arange(0,len(p)) * 10 # lines that passes over the screen

fig,fit = one4all(p, F,0,p_err,"linear",None,"p [mmHg]","F")
m = ufloat(fit.slope,2*fit.stderr)
alpha = m * 2 * kb * T_env * wavelen0 / L
print("Alpha He is:",alpha)

n_He = 1 + alpha * p_nom / (2 * kb * T_nom )
print("n He is:",n_He)
Reg_print(fit)
fig.savefig("fig/plot_He")

#%% Mixture of CO2 and He
print_seciont("Mixture of CO2 and He")
p_env= ufloat(769,1)
'''
first measurments
p = np.array([29,86,159,232,258]) # need to notice units and do conversion to cmHg or other the other way around
p_err = 1
F= np.array([0,9,19,29,34])
p=[155,227,305,402,]
# after jump
p = np.array([551,568,613]) # need to notice units and do conversion to cmHg or other the other way around
p_err = 1
F= np.array([0,3,11])
'''
p = np.array([30,112,142,270,335,398,481,565]) # need to notice units and do conversion to cmHg or other the other way around
p_err = 1
F= np.array([0,6,16,26,36,46,57,68])

#F = np.arange(0,len(p)) * 10 # lines that passes over the screen

fig,fit = one4all(p, F,0,p_err,"linear",None,"p [mmHg]","F")
Reg_print(fit)
m = ufloat(fit.slope,2*fit.stderr)
alpha = m * 2 * kb * T_env * wavelen0 / L
print("Alpha mixture is:",alpha)

n = 1 + alpha * p_nom / (2 * kb * T_nom )
print("n mixture is:",n)

n_He = 1.000036
n_CO2 = 1.00045

CO2overHe = (n_He - n)/(n - n_CO2)
CO2_part = CO2overHe/(1 + CO2overHe) * 100 #percent
He_part =  1/(1 + CO2overHe) * 100 #percent #100 -  CO2_part
print("There is %s%% CO2 and %s%% He in the mixture"%(CO2_part,He_part))
fig.savefig("fig/plot_mixture")