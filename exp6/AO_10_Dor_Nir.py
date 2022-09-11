# LIBRARIES AND MODULES
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy import constants

from uncertainties import unumpy
from uncertainties import ufloat
from uncertainties.umath import *
from uncertainties.unumpy import nominal_values as uval
from uncertainties.unumpy import std_devs as uerr

# FUNCTIONS

# PRETTY PRINT
def PrettyPrint(textA,A,Aerr,Units,Round):
    if Aerr !=0:
        print(" • " + textA + " " + str(round(A,Round)) + " +\- " + str(round(Aerr,Round)) + " " + Units + "\n")
    else:
        print(" • " + textA + str(round(A,Round)) + " " + Units + "\n")
    
# LINEAR REGRESSION    
def Linear(x,a,b):
    return a*x+b

def PlotRegression(x,y,xerror,yerror,xlabel,ylabel,do_reg):
    # Plotting
    plt.figure(dpi=300)
    plt.grid()
    plt.plot(x,y,'m.',label="Data",markersize=10)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    # Errorbars
    if xerror.all() == 0 and yerror.all() != 0:
        plt.errorbar(x,y,xerr=None,yerr=yerror,ls="none",label="Error",color="black")
    elif xerror.all() != 0 and yerror.all() == 0:
        plt.errorbar(x,y,xerr=xerror,yerr=None,ls="none",label="Error",color="black")
    elif xerror.all() != 0 and yerror.all() != 0:
        plt.errorbar(x,y,xerr=xerror,yerr=yerror,ls="none",label="Error",color="black")
    # Linear Regression
    if do_reg == True:
        reg = linregress(x,y)
        slope = reg.slope
        intercept = reg.intercept
        yreg = Linear(x,slope,intercept)
        plt.plot(x,yreg,'c',label="Linear Fit")
        plt.legend()
        return slope,intercept,reg.stderr,reg.intercept_stderr,(reg.rvalue)**2
    else:
        plt.legend()
        return
    
def PrintReg(reg,Round):
    print("R^2=%f :: y = (%f +/- %f)x + (%f +/- %f)"%(round(reg[4],Round),round(reg[0],Round),round(2*reg[2],Round),round(reg[1],Round),round(2*reg[3],Round)))
    
#%%
Noerror = np.array([0])
T = 273 + 23    # Kelvin
T_err = 1       # Kelvin
T = ufloat(T,T_err)

v_theory_at_20_degree = 1160.02 #m/s

T_theory = 293.204 # Kelvin

v = sqrt(293.204/T) * v_theory_at_20_degree
PrettyPrint("v theory :", v.n, v.s, "m/s", 5)

#%%
# PART 4
print("=================PART 4 FF===================\n")
lambd4 = 6328e-10                                 # [m]
f3 = 300e-3

x4_tot = np.array([502,195,126,540,148,203,446,727,262,306,286])*5.2                                      # [um]
x4_tot_err = np.ones(len(x4_tot))*4*5.2                                                                   # [um]
num_x4 = np.array([6,4,2,6,2,2,4,6,2,2,2])
x4_av = x4_tot/num_x4                                                                                     # [um]
freq4 = np.array([2.6446,1.5446,1.9446,2.8446,2.3446,3.2446,3.5446,3.8446,4.1446,4.8446,4.5446])               # [MHz]
freq4_err = np.ones(len(freq4))*1e-5              # [MHz]

reg4 = PlotRegression(x4_av,freq4,x4_tot_err,freq4_err,r"$\Delta x$ $[\mu m]$","Frequency [MHz]",True)

slope =ufloat(reg4[0],2*reg4[2])
v4 = (slope*1e12)*f3*lambd4                     # [m/s]
PrettyPrint("Speed of Sound 4 is: ",v4.n,v4.s,"[m/s]",5)
PrintReg(reg4, 5)


#%%
# PART 3
print("=================PART 3 NF===================\n")

d3_tot = np.array([374,511,210,172,280,290,436,290,299,250])*5.2                                   # [um]
d3_tot_err = np.ones(len(d3_tot))*8*5.2   
d3_tot = unumpy.uarray(d3_tot,d3_tot_err)              # [um]

num_d3 = np.array([8,13,5,5,9,10,6,5,3,3])
d3_av = d3_tot/num_d3                                   # [um]
freq3 = np.array([2.3957,2.8336,2.6414430,3.243188,3.540463,3.84386,1.549243,1.94664,1.129863,1.319723])                         # [MHz]
freq3_err = np.ones(len(freq3))*1e-5                       # [MHz]

x = 1 / d3_av
reg3= PlotRegression(uval(x),freq3,uerr(x),freq3_err,r"$\frac{1}{d}$ $[\frac{1}{\mu m}]$","Frequency [MHz]",True)

slope =ufloat(reg3[0],2*reg3[2])

v3 = 2 * slope                      # [m/s]
PrettyPrint("Speed of Sound 3 is: ",v3.n,v3.s,"[m/s]",5)

PrintReg(reg3, 5)

# HALF WAVELENGTH