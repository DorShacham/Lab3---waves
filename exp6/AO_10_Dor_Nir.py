# LIBRARIES AND MODULES
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy import constants

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
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
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
    
#%%
Noerror = np.array([0])

#%%
# PART 2
print("\n=================PART 2===================\n")

d2_tot = 5               # [um]
d2_tot_err = 0           # [um]
num_d2 = 10
d2_av = d2_tot/10
PrettyPrint("Average d is: ",d2_av,0,"[um]",2)

#%%
# PART 3
print("=================PART 3===================\n")

d3_tot = np.array([15,30,60,90,150])           # [um]
d3_tot_err = np.array([1,1,1,1,1])             # [um]
num_d3 = np.array([1,2,3,4,5])
d3_av = d3_tot/num_d3                         # [um]
freq3 = np.array([2,4,6,8,10])                # [MHz]
freq3_err = np.array([1,1,1,1,1])             # [MHz]

reg3= PlotRegression(d3_av,freq3,Noerror,freq3_err,r"d Average $[\mu m]$","Frequency [MHz]",True)
v3 = 1/(reg3[0]*1e-6)                         # [m/s]
PrettyPrint("Speed of Sound 3 is: ",v3,0,"[m/s]",3)

#%%
# PART 4
print("=================PART 4===================\n")

lambd4 = 6328e-10                                 # [m]
f3 = 300e-3                                       # [m]

x4_tot = np.array([15,30,60,90,150])              # [um]
x4_tot_err = np.array([1,1,1,1,1])                # [um]
x4 = x4_tot/2                                     # [um]
x4_err = x4_tot_err/2                             # [um]
freq4 = np.array([3,6,9,12,15])                   # [Hz]
freq4_err = np.array([1,1,1,1,1])                 # [Hz]
reg4 = PlotRegression(x4,freq4,x4_err,freq4_err,r"$\Delta x$ $[\mu m]$","Frequency [MHz]",True)

v4 = (reg4[0]*1e6)*f3*lambd4                     # [m/s]
PrettyPrint("Speed of Sound 4 is: ",v4,0,"[m/s]",3)


